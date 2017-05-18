'''
Miscellaneous utility functions for natural language processing
'''
import re


def load_tokens_from_pretrained_token_embeddings(parameters):
    count = -1
    tokens = set()
    number_of_loaded_word_vectors = 0
    with open(parameters['token_pretrained_embedding_filepath'], 'r', encoding='UTF-8') as fp:
        for line in fp:
            count += 1
            line = line.strip().split(' ')
            if len(line) == 0:
                continue
            token = line[0]
            tokens.add(token)
            number_of_loaded_word_vectors += 1

    return tokens


def load_pretrained_token_embeddings(parameters):
    file_input = open(parameters['token_pretrained_embedding_filepath'], 'r', encoding='UTF-8')
    count = -1
    token_to_vector = {}
    for cur_line in file_input:
        count += 1
        #if count > 1000:break
        cur_line = cur_line.strip()
        cur_line = cur_line.split(' ')
        if len(cur_line)==0:continue
        token = cur_line[0]
        vector =cur_line[1:]
        token_to_vector[token] = vector
    file_input.close()
    return token_to_vector


def is_token_in_pretrained_embeddings(token, all_pretrained_tokens, parameters):
    return token in all_pretrained_tokens or \
        parameters['check_for_lowercase'] and token.lower() in all_pretrained_tokens or \
        parameters['check_for_digits_replaced_with_zeros'] and re.sub('\d', '0', token) in all_pretrained_tokens or \
        parameters['check_for_lowercase'] and parameters['check_for_digits_replaced_with_zeros'] and re.sub('\d', '0', token.lower()) in all_pretrained_tokens

