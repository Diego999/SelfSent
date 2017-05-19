import os
import numpy as np
import sklearn.metrics
import utils
import random


def train_step(sess, dataset, model, parameters):
    # Perform one iteration

    # Train model: loop over all sequences of training set with shuffling
    sequence_numbers = list(range(len(dataset.token_indices['train'])))
    random.shuffle(sequence_numbers)

    input_token_indices_padded_batches = utils.batch([dataset.token_indices_padded['train'][i] for i in sequence_numbers], parameters['batch_size'])
    input_token_lengths_batches = utils.batch([dataset.token_lengths['train'][i] for i in sequence_numbers], parameters['batch_size'])
    input_label_indices_vector_batches = utils.batch([dataset.label_vector_indices['train'][i] for i in sequence_numbers], parameters['batch_size'])

    total_loss = []
    total_accuracy = []
    step = 0
    for input_token_indices_padded, input_token_lengths, input_label_vector_indices in zip(input_token_indices_padded_batches, input_token_lengths_batches, input_label_indices_vector_batches):
        feed_dict = {
            model.input_token_indices: input_token_indices_padded,
            model.input_token_lengths: input_token_lengths,
            model.input_label_vector_indices: input_label_vector_indices,
            model.dropout_keep_prob: 1.0 - parameters['dropout_rate']
        }

        _, _, loss, accuracy = sess.run([model.train_op, model.global_step, model.loss, model.accuracy], feed_dict)
        total_loss.append(loss)
        total_accuracy.append(accuracy)

        step += parameters['batch_size']
        print('Training {0:.2f}% done'.format(step / len(sequence_numbers) * 100), end='\r', flush=True)

    return total_loss, total_accuracy


def prediction_step(sess, dataset, dataset_type, model, stats_graph_folder, epoch_number, parameters):
    if dataset_type == 'deploy':
        print('Predict labels for the {0} set'.format(dataset_type))
    else:
        print('Evaluate model on the {0} set'.format(dataset_type))
    all_predictions = []
    all_y_true = []
    output_filepath = os.path.join(stats_graph_folder, '{1:03d}_{0}.txt'.format(dataset_type, epoch_number))

    input_token_indices_padded_batches = utils.batch(dataset.token_indices_padded[dataset_type], parameters['batch_size'])
    input_token_lengths_batches = utils.batch(dataset.token_lengths[dataset_type], parameters['batch_size'])
    input_label_indices_vector_batches = utils.batch(dataset.label_vector_indices[dataset_type], parameters['batch_size'])

    step = 0
    for input_token_indices_padded, input_token_lengths, input_label_vector_indices in zip(input_token_indices_padded_batches, input_token_lengths_batches, input_label_indices_vector_batches):
        feed_dict = {
            model.input_token_indices: input_token_indices_padded,
            model.input_token_lengths: input_token_lengths,
            model.input_label_vector_indices: input_label_vector_indices,
            model.dropout_keep_prob: 1.
        }

        predictions, confidences = sess.run([model.yhat, model.confidence], feed_dict)
        predictions, confidences = predictions.tolist(), confidences.tolist()

        assert (len(predictions) == len(input_token_indices_padded))

        prediction_labels = [dataset.index_to_label[prediction] for prediction in predictions]
        gold_labels = [dataset.index_to_label[np.argmax(true_label)] for true_label in input_label_vector_indices]

        with open(output_filepath, 'a', encoding='utf-8') as fp:
            for tokens, pred, conf, gold in zip(input_token_indices_padded, prediction_labels, confidences, gold_labels):
                fp.write("{:1d}\t{:.2f}\t{:1d}\t{}\n".format(pred, 100*conf, gold, ' '.join([dataset.index_to_token[i] for i in tokens if i != dataset.PADDING_TOKEN_INDEX])))

        all_predictions.extend(zip(prediction_labels, confidences))
        all_y_true.extend(gold_labels)

        step += parameters['batch_size']
        print('Predicting {0:.2f}% done'.format(step / len(dataset.token_lengths[dataset_type]) * 100), end='\r', flush=True)

    if dataset_type != 'deploy':
        all_y_pred = [pred[0] for pred in all_predictions]
        classification_report = sklearn.metrics.classification_report(all_y_true, all_y_pred, digits=4, labels=dataset.unique_labels)
        lines = classification_report.split('\n')
        classification_report = ['Accuracy: {:05.2f}%'.format(sklearn.metrics.accuracy_score(all_y_true, all_y_pred) * 100)]
        for line in lines[2: (len(lines) - 1)]:
            new_line = []
            t = line.strip().replace('avg / total', 'micro-avg').split()
            if len(t) < 2: continue
            new_line.append(('        ' if t[0].isdigit() else '')+ t[0])
            new_line += ['{:05.2f}'.format(float(x) * 100) for x in t[1: len(t) - 1]]
            new_line.append(t[-1])
            classification_report.append('\t'.join(new_line))
        classification_report = '\n'.join(classification_report)
        print('\n\n' + classification_report + '\n', flush=True)
        with open(output_filepath + '_evaluation.txt', 'a', encoding='utf-8') as fp:
            fp.write(classification_report)

    return all_predictions, all_y_true, output_filepath


def predict_labels(sess, model, parameters, dataset, epoch_number, stats_graph_folder, dataset_filepaths):
    # Predict labels using trained model
    y_pred = {}
    y_true = {}
    output_filepaths = {}
    for dataset_type in ['train', 'valid', 'test', 'deploy']:
        if dataset_type not in dataset_filepaths.keys():
            continue
        prediction_output = prediction_step(sess, dataset, dataset_type, model, stats_graph_folder, epoch_number, parameters)
        y_pred[dataset_type], y_true[dataset_type], output_filepaths[dataset_type] = prediction_output
    return y_pred, y_true, output_filepaths


def restore_model_parameters_from_pretrained_model(parameters, dataset, sess, model, model_saver):
    # TODO
    '''
    pretraining_dataset = pickle.load(open(os.path.join(parameters['pretrained_model_folder'], 'dataset.pickle'), 'rb'))
    pretrained_model_checkpoint_filepath = os.path.join(parameters['pretrained_model_folder'], 'model.ckpt')

    # Assert that the label sets are the same
    # Test set should have the same label set as the pretrained dataset
    assert pretraining_dataset.index_to_label == dataset.index_to_label

    # Assert that the model hyperparameters are the same
    pretraining_parameters = main.load_parameters(parameters_filepath=os.path.join(parameters['pretrained_model_folder'], 'parameters.ini'), verbose=False)[0]
    for name in ['use_character_lstm', 'character_embedding_dimension', 'character_lstm_hidden_state_dimension',
                 'token_embedding_dimension', 'token_lstm_hidden_state_dimension', 'use_crf']:
        if parameters[name] != pretraining_parameters[name]:
            print("Parameters of the pretrained model:")
            pprint(pretraining_parameters)
            raise AssertionError(
                "The parameter {0} ({1}) is different from the pretrained model ({2}).".format(name, parameters[name],
                                                                                               pretraining_parameters[
                                                                                                   name]))

    # If the token and character mappings are exactly the same
    if pretraining_dataset.index_to_token == dataset.index_to_token and pretraining_dataset.index_to_character == dataset.index_to_character:

        # Restore the pretrained model
        model_saver.restore(sess,
                            pretrained_model_checkpoint_filepath)  # Works only when the dimensions of tensor variables are matched.

    # If the token and character mappings are different between the pretrained model and the current model
    else:

        # Resize the token and character embedding weights to match them with the pretrained model (required in order to restore the pretrained model)
        utils_tf.resize_tensor_variable(sess, model.character_embedding_weights, [pretraining_dataset.alphabet_size,
                                                                                  parameters[
                                                                                      'character_embedding_dimension']])
        utils_tf.resize_tensor_variable(sess, model.token_embedding_weights,
                                        [pretraining_dataset.vocabulary_size, parameters['token_embedding_dimension']])

        # Restore the pretrained model
        model_saver.restore(sess,
                            pretrained_model_checkpoint_filepath)  # Works only when the dimensions of tensor variables are matched.

        # Get pretrained embeddings
        character_embedding_weights, token_embedding_weights = sess.run(
            [model.character_embedding_weights, model.token_embedding_weights])

        # Restore the sizes of token and character embedding weights
        utils_tf.resize_tensor_variable(sess, model.character_embedding_weights,
                                        [dataset.alphabet_size, parameters['character_embedding_dimension']])
        utils_tf.resize_tensor_variable(sess, model.token_embedding_weights,
                                        [dataset.vocabulary_size, parameters['token_embedding_dimension']])

        # Re-initialize the token and character embedding weights
        sess.run(tf.variables_initializer([model.character_embedding_weights, model.token_embedding_weights]))

        # Load embedding weights from pretrained token embeddings first
        model.load_pretrained_token_embeddings(sess, dataset, parameters)

        # Load embedding weights from pretrained model
        model.load_embeddings_from_pretrained_model(sess, dataset, pretraining_dataset, token_embedding_weights,
                                                    embedding_type='token')
        model.load_embeddings_from_pretrained_model(sess, dataset, pretraining_dataset, character_embedding_weights,
                                                    embedding_type='character')

        del pretraining_dataset
        del character_embedding_weights
        del token_embedding_weights

    # Get transition parameters
    transition_params_trained = sess.run(model.transition_parameters)

    if not parameters['reload_character_embeddings']:
        sess.run(tf.variables_initializer([model.character_embedding_weights]))
    if not parameters['reload_character_lstm']:
        sess.run(tf.variables_initializer(model.character_lstm_variables))
    if not parameters['reload_token_embeddings']:
        sess.run(tf.variables_initializer([model.token_embedding_weights]))
    if not parameters['reload_token_lstm']:
        sess.run(tf.variables_initializer(model.token_lstm_variables))
    if not parameters['reload_feedforward']:
        sess.run(tf.variables_initializer(model.feedforward_variables))
    if not parameters['reload_crf']:
        sess.run(tf.variables_initializer(model.crf_variables))

    return transition_params_trained
    '''
    return None


