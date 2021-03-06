import json
import os
import time
import pickle
import utils_nlp
import utils
import random
import re


class Dataset(object):
    """A class for handling data sets."""

    def __init__(self, name='', verbose=False, debug=False):
        self.name = name
        self.verbose = verbose
        self.debug = debug

    def load_deploy(self, dataset_filepath, parameters, annotator):
        _, tokens, _, _ = self._parse_dataset(dataset_filepath, annotator, force_preprocessing=parameters['do_split'], limit=self.max_tokens)
        self.tokens['deploy'] = tokens

        # Map tokens and labels to their indices
        self.token_indices['deploy'] = []
        self.token_lengths['deploy'] = []
        self.token_indices_padded['deploy'] = []

        # Tokens
        for token_sequence in tokens:
            self.token_indices['deploy'].append([self.token_to_index[token] for token in token_sequence])
            self.token_lengths['deploy'].append(len(token_sequence))

        # Pad tokens
        self.token_indices_padded['deploy'] = []
        self.token_indices_padded['deploy'] = [utils.pad_list(temp_token_indices, self.max_tokens, self.PADDING_TOKEN_INDEX) for temp_token_indices in self.token_indices['deploy']]

        self.labels['deploy'] = []
        self.label_vector_indices['deploy'] = []

    def load_dataset(self, dataset_filepaths, parameters, annotator):
        '''
            dataset_filepaths : dictionary with keys 'train', 'valid', 'test'
        '''
        start_time = time.time()
        print('Load dataset... ', end='', flush=True)

        if parameters['do_split']:
            dataset_filepaths = self._do_split(parameters)

        all_pretrained_tokens = []
        if parameters['token_pretrained_embedding_filepath'] != '':
            all_pretrained_tokens = utils_nlp.load_tokens_from_pretrained_token_embeddings(parameters)
        if self.verbose: print("len(all_pretrained_tokens): {0}".format(len(all_pretrained_tokens)))

        # Load pretraining dataset to ensure that index to label is compatible to the pretrained model,
        #   and that token embeddings that are learned in the pretrained model are loaded properly.
        all_tokens_in_pretraining_dataset = []

        self.UNK_TOKEN_INDEX = 0
        self.PADDING_TOKEN_INDEX = 1
        self.tokens_mapped_to_unk = []
        self.UNK = '_UNK_'
        self.PAD = '_PAD_'
        self.unique_labels = []
        labels = {}
        tokens = {}
        token_count = {}
        label_count = {}

        self.max_tokens = -1
        # Look for max length
        for dataset_type in ['train', 'valid', 'test']:
            max_tokens = self._find_max_length(dataset_filepaths.get(dataset_type, None), annotator, force_preprocessing=parameters['do_split'])
            if parameters['max_length_sentence'] == -1:
                self.max_tokens = max(self.max_tokens, max_tokens)
            else:
                if self.max_tokens == -1:
                    self.max_tokens = max_tokens
                self.max_tokens = min(parameters['max_length_sentence'], self.max_tokens)

        for dataset_type in ['train', 'valid', 'test']:
            labels[dataset_type], tokens[dataset_type], token_count[dataset_type], label_count[dataset_type] = self._parse_dataset(dataset_filepaths.get(dataset_type, None), annotator, force_preprocessing=parameters['do_split'], limit=self.max_tokens)

            if self.verbose: print("dataset_type: {0}".format(dataset_type))
            if self.verbose: print("len(token_count[dataset_type]): {0}".format(len(token_count[dataset_type])))

        token_count['all'] = {}
        for token in list(token_count['train'].keys()) + list(token_count['valid'].keys()) + list(token_count['test'].keys()):
            token_count['all'][token] = token_count['train'].get(token, 0) + token_count['valid'].get(token, 0) + token_count['test'].get(token, 0)

        for dataset_type in dataset_filepaths.keys():
            if self.verbose: print("dataset_type: {0}".format(dataset_type))
            if self.verbose: print("len(token_count[dataset_type]): {0}".format(len(token_count[dataset_type])))

        label_count['all'] = {}
        for character in list(label_count['train'].keys()) + list(label_count['valid'].keys()) + list(label_count['test'].keys()):
            label_count['all'][character] = label_count['train'].get(character, 0) + label_count['valid'].get(character, 0) + label_count['test'].get(character, 0)

        token_count['all'] = utils.order_dictionary(token_count['all'], 'value_key', reverse=True)
        label_count['all'] = utils.order_dictionary(label_count['all'], 'key', reverse=False)

        token_to_index = {}
        token_to_index[self.UNK] = self.UNK_TOKEN_INDEX
        token_to_index[self.PAD] = self.PADDING_TOKEN_INDEX
        iteration_number = 0
        number_of_unknown_tokens = 0

        if self.verbose: print("parameters['remap_unknown_tokens_to_unk']: {0}".format(parameters['remap_unknown_tokens_to_unk']))
        if self.verbose: print("len(token_count['train'].keys()): {0}".format(len(token_count['train'].keys())))

        for token, count in token_count['all'].items():
            if iteration_number == self.UNK_TOKEN_INDEX:
                iteration_number += 1
            if iteration_number == self.PADDING_TOKEN_INDEX:
                iteration_number += 1

            if parameters['remap_unknown_tokens_to_unk'] and (token_count['train'].get(token, 0) == 0 or parameters['load_only_pretrained_token_embeddings']) and not utils_nlp.is_token_in_pretrained_embeddings(token, all_pretrained_tokens, parameters) and token not in all_tokens_in_pretraining_dataset:
                if self.verbose: print("token: {0}".format(token))
                if self.verbose: print("token.lower(): {0}".format(token.lower()))
                if self.verbose: print("re.sub('\d', '0', token.lower()): {0}".format(re.sub('\d', '0', token.lower())))
                token_to_index[token] = self.UNK_TOKEN_INDEX
                number_of_unknown_tokens += 1
                self.tokens_mapped_to_unk.append(token)
            else:
                token_to_index[token] = iteration_number
                iteration_number += 1

        if self.verbose: print("number_of_unknown_tokens: {0}".format(number_of_unknown_tokens))

        infrequent_token_indices = []
        for token, count in token_count['train'].items():
            if 0 < count <= parameters['remap_to_unk_count_threshold']:
                infrequent_token_indices.append(token_to_index[token])

        if self.verbose: print("len(token_count['train']): {0}".format(len(token_count['train'])))
        if self.verbose: print("len(infrequent_token_indices): {0}".format(len(infrequent_token_indices)))

        label_to_index = {}
        iteration_number = 0
        for label, count in label_count['all'].items():
            label_to_index[label] = iteration_number
            iteration_number += 1
            self.unique_labels.append(label)

        if self.verbose: print('self.unique_labels: {0}'.format(self.unique_labels))
        if self.verbose: print('token_count[\'train\'][0:10]: {0}'.format(list(token_count['train'].items())[0:10]))

        token_to_index = utils.order_dictionary(token_to_index, 'value', reverse=False)

        if self.verbose: print('token_to_index: {0}'.format(token_to_index))

        index_to_token = utils.reverse_dictionary(token_to_index)

        if parameters['remap_unknown_tokens_to_unk'] == 1:
            index_to_token[self.UNK_TOKEN_INDEX] = self.UNK
        index_to_token[self.PADDING_TOKEN_INDEX] = self.PAD

        if self.verbose: print('index_to_token: {0}'.format(index_to_token))
        if self.verbose: print('label_count[\'train\']: {0}'.format(label_count['train']))

        label_to_index = utils.order_dictionary(label_to_index, 'value', reverse=False)

        if self.verbose: print('label_to_index: {0}'.format(label_to_index))

        index_to_label = utils.reverse_dictionary(label_to_index)

        if self.verbose: print('index_to_label: {0}'.format(index_to_label))
        if self.verbose: print('labels[\'train\'][0:10]: {0}'.format(labels['train'][0:10]))
        if self.verbose: print('tokens[\'train\'][0:10]: {0}'.format(tokens['train'][0:10]))

        # Map tokens and labels to their indices
        token_indices = {}
        label_indices = {}
        token_lengths = {}
        token_indices_padded = {}
        for dataset_type in dataset_filepaths.keys():
            token_indices[dataset_type] = []
            token_lengths[dataset_type] = []
            token_indices_padded[dataset_type] = []

            # Tokens
            for token_sequence in tokens[dataset_type]:
                token_indices[dataset_type].append([token_to_index[token] for token in token_sequence])
                token_lengths[dataset_type].append(len(token_sequence))

            # Labels
            label_indices[dataset_type] = []
            for label in labels[dataset_type]:
                label_indices[dataset_type].append(label_to_index[label])

        # Pad tokens
        for dataset_type in dataset_filepaths.keys():
            token_indices_padded[dataset_type] = []
            token_indices_padded[dataset_type] = [utils.pad_list(temp_token_indices, self.max_tokens, self.PADDING_TOKEN_INDEX) for temp_token_indices in token_indices[dataset_type]]

        if self.verbose: print('token_lengths[\'train\'][0:10]: {0}'.format(token_lengths['train'][0:10]))
        if self.verbose: print('token_indices[\'train\'][0][0:10]: {0}'.format(token_indices['train'][0][0:10]))
        if self.verbose: print('token_indices_padded[\'train\'][0][0:10]: {0}'.format(token_indices_padded['train'][0][0:10]))
        if self.verbose: print('label_indices[\'train\'][0:10]: {0}'.format(label_indices['train'][0:10]))

        self.token_to_index = token_to_index
        self.index_to_token = index_to_token
        self.token_indices = token_indices
        self.label_indices = label_indices
        self.token_indices_padded = token_indices_padded
        self.token_lengths = token_lengths
        self.tokens = tokens
        self.labels = labels
        self.index_to_label = index_to_label
        self.label_to_index = label_to_index

        if self.verbose: print("len(self.token_to_index): {0}".format(len(self.token_to_index)))
        if self.verbose: print("len(self.index_to_token): {0}".format(len(self.index_to_token)))

        self.number_of_classes = max(self.index_to_label.keys()) + 1
        self.vocabulary_size = max(self.index_to_token.keys()) + 1

        if self.verbose: print("self.number_of_classes: {0}".format(self.number_of_classes))
        if self.verbose: print("self.vocabulary_size: {0}".format(self.vocabulary_size))

        self.infrequent_token_indices = infrequent_token_indices

        # Binarize label
        label_vector_indices = {}
        for dataset_type, labels in label_indices.items():
            label_vector_indices[dataset_type] = []
            for label in labels:
                label_vector_indices[dataset_type].append(utils.convert_one_hot(label, self.number_of_classes))
        self.label_vector_indices = label_vector_indices

        elapsed_time = time.time() - start_time
        print('done ({0:.2f} seconds)'.format(elapsed_time))

    def _parse_dataset(self, dataset_filepath, annotator, force_preprocessing=False, limit=-1):
        token_count = {}
        label_count = {}

        tokens = []
        labels = []
        if dataset_filepath != '':
            dataset_filepath_pickle = dataset_filepath.replace('json', 'pickle')
            if os.path.isfile(dataset_filepath_pickle) and not force_preprocessing:
                with open(dataset_filepath_pickle, 'rb') as fp:
                    return pickle.load(fp)
            else:
                with open(dataset_filepath, 'r', encoding='utf-8') as fp:
                    for sample in fp:
                        parsed_data = json.loads(sample, encoding='utf-8')

                        token_sequence = []
                        for token_found in annotator.parse_doc(parsed_data['text'])['sentences']:
                            token_sequence += token_found['tokens']

                        if limit != -1:
                            token_sequence = token_sequence[:limit]

                        tokens.append(token_sequence)
                        if 'stars' in parsed_data:
                            labels.append(int(parsed_data['stars']))

                        for token in token_sequence:
                            if token not in token_count:
                                token_count[token] = 0
                            token_count[token] += 1

                        if len(labels) > 0:
                            if labels[-1] not in label_count:
                                label_count[labels[-1]] = 0
                            label_count[labels[-1]] += 1

                with open(dataset_filepath_pickle, 'wb') as fp:
                    obj = [labels, tokens, token_count, label_count]
                    pickle.dump(obj, fp)

        return labels, tokens, token_count, label_count

    def _find_max_length(self, dataset_filepath, annotator, force_preprocessing=False):
        max_length = 0
        if dataset_filepath:
            dataset_filepath_pickle = dataset_filepath.replace('.json', '_max.pickle')
            if os.path.isfile(dataset_filepath_pickle) and not force_preprocessing:
                with open(dataset_filepath_pickle, 'rb') as fp:
                    return pickle.load(fp)
            else:
                with open(dataset_filepath, 'r', encoding='utf-8') as fp:
                    for sample in fp:
                        parsed_data = json.loads(sample, encoding='utf-8')
                        max_length = max(max_length, sum([len(sentence['tokens']) for sentence in annotator.parse_doc(parsed_data['text'])['sentences']]))
                    with open(dataset_filepath_pickle, 'wb') as fp:
                        pickle.dump(max_length, fp)

        return max_length

    def _do_split(self, parameters):
        data = []
        with open(os.path.join(parameters['dataset_folder'], 'all.json'), 'r', encoding='utf-8') as fp:
            for line in fp:
                data.append(line)

        random.shuffle(data)
        dataset_filepaths = {}
        for dataset_type in ['train', 'valid', 'test']:
            size = parameters[dataset_type + '_size']
            data_set = data[:size]
            data = data[size:]

            dataset_filepaths[dataset_type] = os.path.join(parameters['dataset_folder'], dataset_type + '.json')
            with open(dataset_filepaths[dataset_type], 'w', encoding='utf-8') as fp:
                for d in data_set:
                    fp.write(d)

            # Removed preprocessed file
            pickle_file = dataset_filepaths[dataset_type].replace('json', 'pickle')
            if os.path.isfile(pickle_file):
                os.remove(pickle_file)

        return dataset_filepaths
