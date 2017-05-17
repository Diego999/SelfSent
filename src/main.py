import os
import tensorflow as tf
import warnings
import configparser
import random
import utils
import distutils.util
import pprint
import dataset as ds
import sys
import time
import copy
from self_sent import SelfSent
import pickle
import dill


warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
print('SelfSent version: {0}'.format('1.0-dev'))
print('TensorFlow version: {0}'.format(tf.__version__))


def load_parameters(parameters_filepath=os.path.join('.', 'parameters.ini'), verbose=True):
    '''
    Load parameters from the ini file, and ensure that each parameter is cast to the correct type
    '''
    conf_parameters = configparser.ConfigParser()
    conf_parameters.read(parameters_filepath)
    nested_parameters = utils.convert_configparser_to_dictionary(conf_parameters)
    parameters = {}
    for k, v in nested_parameters.items():
        parameters.update(v)

    for k, v in parameters.items():
        # If the value is a list delimited with a comma, choose one element at random.
        # Ensure that each parameter is cast to the correct type
        if k in ['seed', 'train_size', 'valid_size', 'test_size', 'remap_to_unk_count_threshold', 'token_embedding_dimension', 'number_of_cpu_threads', 'number_of_gpus', 'lstm_hidden_state_dimension', 'batch_size', 'da', 'r', 'mlp_hidden_layer_1_units', 'mlp_hidden_layer_2_units']:
            parameters[k] = int(v)
        elif k in ['beta']:
            parameters[k] = float(v)
        elif k in ['freeze_token_embeddings', 'do_split', 'remap_unknown_tokens_to_unk', 'verbose', 'debug', 'use_pretrained_model', 'load_only_pretrained_token_embeddings', 'check_for_lowercase', 'check_for_digits_replaced_with_zeros']:
            parameters[k] = distutils.util.strtobool(v)

    if verbose:
        pprint.pprint(parameters)

    return parameters, conf_parameters


def get_valid_dataset_filepaths(parameters):
    dataset_filepaths = {}
    dataset_brat_folders = {}
    for dataset_type in ['train', 'valid', 'test', 'deploy']:
        dataset_filepaths[dataset_type] = os.path.join(parameters['dataset_folder'], '{0}.json'.format(dataset_type))

        # Json files exists
        if os.path.isfile(dataset_filepaths[dataset_type]) and os.path.getsize(dataset_filepaths[dataset_type]) > 0:
            dataset_filepaths[dataset_type] = dataset_filepaths[dataset_type]
        else:
            dataset_filepaths[dataset_type] = None

    return dataset_filepaths


def main():
    file_params = 'parameters.ini'
    if len(sys.argv) > 1 and '.ini' in sys.argv[1]:
        file_params = sys.argv[1]

    # Load config
    parameters, conf_parameters = load_parameters(parameters_filepath=os.path.join('.', file_params))
    dataset_filepaths = get_valid_dataset_filepaths(parameters)
    #check_parameter_compatiblity(parameters, dataset_filepaths)

    if parameters['seed'] != -1:
        random.seed(parameters['seed'])

    # Load dataset
    dataset = ds.Dataset(verbose=parameters['verbose'], debug=parameters['debug'])
    dataset.load_dataset(dataset_filepaths, parameters)

    # Create graph and session
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            intra_op_parallelism_threads=parameters['number_of_cpu_threads'],
            inter_op_parallelism_threads=parameters['number_of_cpu_threads'],
            device_count={'CPU': 1, 'GPU': parameters['number_of_gpus']},
            allow_soft_placement=True, # automatically choose an existing and supported device to run the operations in case the specified one doesn't exist
            log_device_placement=False
            )

        sess = tf.Session(config=session_conf)

        with sess.as_default():
            if parameters['seed'] != -1:
                tf.set_random_seed(parameters['seed'])

                # Initialize and save execution details
                start_time = time.time()
                experiment_timestamp = utils.get_current_time_in_miliseconds()

                results = {}
                results['epoch'] = {}
                results['execution_details'] = {}
                results['execution_details']['train_start'] = start_time
                results['execution_details']['time_stamp'] = experiment_timestamp
                results['execution_details']['early_stop'] = False
                results['execution_details']['keyboard_interrupt'] = False
                results['execution_details']['num_epochs'] = 0
                results['model_options'] = copy.copy(parameters)

                dataset_name = utils.get_basename_without_extension(parameters['dataset_folder'])
                model_name = '{0}_{1}'.format(dataset_name, results['execution_details']['time_stamp'])

                output_folder = os.path.join('..', 'output')
                utils.create_folder_if_not_exists(output_folder)

                stats_graph_folder = os.path.join(output_folder, model_name)  # Folder where to save graphs
                utils.create_folder_if_not_exists(stats_graph_folder)
                model_folder = os.path.join(stats_graph_folder, 'model')
                utils.create_folder_if_not_exists(model_folder)

                with open(os.path.join(model_folder, file_params), 'w') as parameters_file:
                    conf_parameters.write(parameters_file)

                tensorboard_log_folder = os.path.join(stats_graph_folder, 'tensorboard_logs')
                utils.create_folder_if_not_exists(tensorboard_log_folder)
                tensorboard_log_folders = {}
                for dataset_type in dataset_filepaths.keys():
                    tensorboard_log_folders[dataset_type] = os.path.join(stats_graph_folder, 'tensorboard_logs', dataset_type)
                    utils.create_folder_if_not_exists(tensorboard_log_folders[dataset_type])

                # TODO
                #dill.dump(dataset, open(os.path.join(model_folder, 'dataset.pickle'), 'wb'))

                # Instantiate the model
                # graph initialization should be before FileWriter, otherwise the graph will not appear in TensorBoard
                model = SelfSent(dataset, parameters)

if __name__ == "__main__":
    main()
