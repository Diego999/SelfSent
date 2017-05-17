import tensorflow as tf
import utils_tf


def bidirectional_LSTM(input, hidden_state_dimension, initializer, batch_size, sequence_length):
    with tf.variable_scope("bidirectional_LSTM"):
        lstm_cell = {}
        initial_state = {}
        for direction in ["forward", "backward"]:
            with tf.variable_scope(direction):
                # LSTM cell
                lstm_cell[direction] = tf.contrib.rnn.CoupledInputForgetGateLSTMCell(hidden_state_dimension, forget_bias=1.0, initializer=initializer, state_is_tuple=True)
                # initial state: http://stackoverflow.com/questions/38441589/tensorflow-rnn-initial-state
                initial_cell_state = tf.get_variable("initial_cell_state", shape=[1, hidden_state_dimension], dtype=tf.float32, initializer=initializer)
                initial_output_state = tf.get_variable("initial_output_state", shape=[1, hidden_state_dimension], dtype=tf.float32, initializer=initializer)
                c_states = tf.tile(initial_cell_state, tf.stack([batch_size, 1]))
                h_states = tf.tile(initial_output_state, tf.stack([batch_size, 1]))
                initial_state[direction] = tf.contrib.rnn.LSTMStateTuple(c_states, h_states)

        # sequence_length must be provided for tf.nn.bidirectional_dynamic_rnn due to internal bug
        (outputs_forward, outputs_backward), final_states = tf.nn.bidirectional_dynamic_rnn(lstm_cell["forward"],
                                                                    lstm_cell["backward"],
                                                                    input,
                                                                    dtype=tf.float32,
                                                                    sequence_length=sequence_length,
                                                                    initial_state_fw=initial_state["forward"],
                                                                    initial_state_bw=initial_state["backward"])

        output = tf.concat([outputs_forward, outputs_backward], axis=2, name='output_sequence')

    return output


class SelfSent(object):

    def __init__(self, dataset, parameters):
        self.verbose = parameters['verbose']
        self.dataset = dataset

        # Placeholders for input, output and dropout
        self.input_token_indices = tf.placeholder(tf.int32, [parameters['batch_size'], self.dataset.max_tokens], name="input_token_indices")
        self.input_token_lengths = tf.placeholder(tf.int32, [parameters['batch_size']], name="input_token_lengths")
        self.input_label_indices = tf.placeholder(tf.float32, [parameters['batch_size'], dataset.number_of_classes], name="input_label_indices")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Internal parameters
        initializer = tf.contrib.layers.xavier_initializer()

        # Token embedding layer
        with tf.variable_scope("token_embedding"):
            self.token_embedding_weights = tf.get_variable("token_embedding_weights", shape=[dataset.vocabulary_size, parameters['token_embedding_dimension']], initializer=initializer, trainable=not parameters['freeze_token_embeddings'])
            token_lstm_input = tf.nn.embedding_lookup(self.token_embedding_weights, self.input_token_indices)
            if self.verbose: print("token_lstm_input: {0}".format(token_lstm_input))
            utils_tf.variable_summaries(self.token_embedding_weights)

        # Add dropout
        with tf.variable_scope("dropout"):
            token_lstm_input_drop = tf.nn.dropout(token_lstm_input, self.dropout_keep_prob, name='token_lstm_input_drop')
            if self.verbose: print("token_lstm_input_drop: {0}".format(token_lstm_input_drop))

        # BiLSTM
        with tf.variable_scope("token_lstm") as vs:
            H = bidirectional_LSTM(token_lstm_input_drop, parameters['lstm_hidden_state_dimension'], initializer, parameters['batch_size'], self.input_token_lengths)
            if self.verbose: print("H: {0}".format(H))
            self.token_lstm_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=vs.name)

        # Because we use batch, H is a 3D matrix while Ws1 and Ws2 are 2D matrix.
        # To simplify the computation:
        # M = A * H = softmax(Ws2 * tanh(Ws1 * H^T)) * H
        # M^T = H^T * A^T = H^T * softmax(tanh(H * Ws1^T) * Ws2^T)

        # Attention
        with tf.variable_scope("attention") as vs:
            Ws1 = tf.get_variable("Ws1", shape=[2*parameters['lstm_hidden_state_dimension'], parameters['da']], initializer=initializer)
            H_reshaped = tf.reshape(H, [-1, 2*parameters['lstm_hidden_state_dimension']], name='H_reshaped')
            tanh_ws1_time_H = tf.nn.tanh(tf.matmul(H_reshaped, Ws1), name="tanh_ws1_time_H")

            Ws2 = tf.get_variable("Ws2", shape=[parameters['da'], parameters['r']], initializer=initializer)
            tanh_ws1_time_H_and_time_Ws2 = tf.matmul(tanh_ws1_time_H, Ws2, name="tanh_ws1_time_H_and_time_Ws2")

            # The final softmax should be applied for the dimension corresponding to the tokens
            A_T = tf.nn.softmax(tf.reshape(tanh_ws1_time_H_and_time_Ws2, shape=[parameters['batch_size'], self.dataset.max_tokens, parameters['r']], name="A_T_no_softmax"), dim=1, name="A_T")
            self.attention_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=vs.name)

        # Apply Attention
        with tf.variable_scope("apply_attention"):
            H_T = tf.transpose(H, perm=[0, 2, 1], name="H_T")
            M_T = tf.matmul(H_T, A_T, name="M_T_no_transposed")

        # Compute penalization term
        with tf.variable_scope("penalization_term"):
            A = tf.transpose(A_T, perm=[0, 2, 1], name="A")
            AA_T = tf.matmul(A, A_T, name="AA_T")
            identity = tf.reshape(tf.tile(tf.diag(tf.ones([parameters['r']]), name="diag_identity"), [parameters['batch_size'], 1], name="tile_identity"), [parameters['batch_size'], parameters['r'], parameters['r']], name="identity")
            penalized_term = tf.square(tf.norm(AA_T - identity, ord='euclidean', axis=[1, 2], name="frobenius_norm"), name="penalized_term")