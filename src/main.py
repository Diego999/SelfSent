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
        if k in ['seed', 'train_size', 'valid_size', 'test_size', 'remap_to_unk_count_threshold', 'token_embedding_dimension']:
            parameters[k] = int(v)
        elif k in ['']:
            parameters[k] = float(v)
        elif k in ['do_split', 'remap_unknown_tokens_to_unk', 'verbose', 'debug', 'use_pretrained_model', 'load_only_pretrained_token_embeddings', 'check_for_lowercase', 'check_for_digits_replaced_with_zeros']:
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

if __name__ == "__main__":
    main()
