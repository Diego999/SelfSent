import json
import pprint
import os
import time
import pickle
import utils_nlp
import utils
import random
import stanford_corenlp_pywrapper
import re


class Dataset(object):
    """A class for handling data sets."""

    def __init__(self, name='', verbose=False, debug=False):
        self.name = name
        self.verbose = verbose
        self.debug = debug

    def load_dataset(self, dataset_filepaths, parameters):
        '''
            dataset_filepaths : dictionary with keys 'train', 'valid', 'test', 'deploy'
        '''
        start_time = time.time()
        self.annotators = stanford_corenlp_pywrapper.CoreNLP(configdict={'annotators': 'tokenize, ssplit', 'ssplit.eolonly': True}, corenlp_jars=[parameters['stanford_folder'] + '/*'])
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
        if parameters['use_pretrained_model']:
            pretraining_dataset = pickle.load(open(os.path.join(parameters['pretrained_model_folder'], 'dataset.pickle'), 'rb'))
            all_tokens_in_pretraining_dataset = pretraining_dataset.index_to_token.values()

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
        for dataset_type in ['train', 'valid', 'test', 'deploy']:
            labels[dataset_type], tokens[dataset_type], token_count[dataset_type], label_count[dataset_type] = self._parse_dataset(dataset_filepaths.get(dataset_type, None))

            if self.verbose: print("dataset_type: {0}".format(dataset_type))
            if self.verbose: print("len(token_count[dataset_type]): {0}".format(len(token_count[dataset_type])))

        token_count['all'] = {}
        for token in list(token_count['train'].keys()) + list(token_count['valid'].keys()) + list(token_count['test'].keys()) + list(token_count['deploy'].keys()):
            token_count['all'][token] = token_count['train'].get(token, 0) + token_count['valid'].get(token, 0) + token_count['test'].get(token, 0) + token_count['deploy'].get(token, 0)

        for dataset_type in dataset_filepaths.keys():
            if self.verbose: print("dataset_type: {0}".format(dataset_type))
            if self.verbose: print("len(token_count[dataset_type]): {0}".format(len(token_count[dataset_type])))

        label_count['all'] = {}
        for character in list(label_count['train'].keys()) + list(label_count['valid'].keys()) + list(label_count['test'].keys()) + list(label_count['deploy'].keys()):
            label_count['all'][character] = label_count['train'].get(character, 0) + label_count['valid'].get(character, 0) + label_count['test'].get(character, 0) + label_count['deploy'].get(character, 0)

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

        if parameters['use_pretrained_model']:
            self.unique_labels = sorted(list(pretraining_dataset.label_to_index.keys()))
            # Make sure labels are compatible with the pretraining dataset.
            for label in label_count['all']:
                if label not in pretraining_dataset.label_to_index:
                    raise AssertionError("The label {0} does not exist in the pretraining dataset. ".format(label) +
                                         "Please ensure that only the following labels exist in the dataset: {0}".format(', '.join(self.unique_labels)))
            label_to_index = pretraining_dataset.label_to_index.copy()
        else:
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

            if len(token_lengths[dataset_type]) > 0:
                longest_token_length_in_sequence = max(token_lengths[dataset_type])
                token_indices_padded[dataset_type] = [utils.pad_list(temp_token_indices, longest_token_length_in_sequence, self.PADDING_TOKEN_INDEX) for temp_token_indices in token_indices[dataset_type]]

            label_indices[dataset_type] = []
            for label in labels[dataset_type]:
                label_indices[dataset_type].append(label_to_index[label])

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

        elapsed_time = time.time() - start_time
        print('done ({0:.2f} seconds)'.format(elapsed_time))

    def _parse_dataset(self, dataset_filepath):
        token_count = {}
        label_count = {}

        tokens = []
        labels = []
        if dataset_filepath:
            dataset_filepath_pickle = dataset_filepath.replace('json', 'pickle')
            if os.path.isfile(dataset_filepath_pickle):
                with open(dataset_filepath_pickle, 'rb') as fp:
                    return pickle.load(fp)
            else:
                with open(dataset_filepath, 'r', encoding='utf-8') as fp:
                    for sample in fp:
                        parsed_data = json.loads(sample, encoding='utf-8')

                        token_sequence = []
                        for token_found in self.annotators.parse_doc(parsed_data['text'])['sentences']:
                            token_sequence += token_found['tokens']
                        tokens.append(token_sequence)
                        labels.append(int(parsed_data['stars']))

                        for token in token_sequence:
                            if token not in token_count:
                                token_count[token] = 0
                            token_count[token] += 1

                        if labels[-1] not in label_count:
                            label_count[labels[-1]] = 0
                        label_count[labels[-1]] += 1

                with open(dataset_filepath_pickle, 'wb') as fp:
                    obj = [labels, tokens, token_count, label_count]
                    pickle.dump(obj, fp)

        return labels, tokens, token_count, label_count

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
                fp.write(''.join(data_set))

            # Removed preprocessed file
            pickle_file = dataset_filepaths[dataset_type].replace('json', 'pickle')
            if os.path.isfile(pickle_file):
                os.remove(pickle_file)

        return dataset_filepaths
