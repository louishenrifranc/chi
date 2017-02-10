import tensorflow_fold as td
from tensorflow.contrib import layers

import tensorflow as tf
import numpy as np
import chi

FLAGS = tf.app.flags.FLAGS
FLAGS.DEFINE_integer('embedding_size', 32, 'embedding size of word in reviews')
FLAGS.DEFINE_integer('batch_size', 64, 'batch size')
FLAGS.DEFINE_integer('max_sentence_length', 128, 'max sentence length')
FLAGS.DEFINE_integer('hidden_size', 32, 'hidden size of rnn')


@chi.experiment(local_dependencies=[chi])
def model():
    """
    Helper functions
    ----------------
    """

    def _length_sequence(sentences):
        used = tf.sign(tf.reduce_sum(tf.abs(sentences), reduction_indices=2))
        length = tf.reduce_sum(used, reduction_indices=1)
        return tf.cast(length, tf.int32)

    def _get_last_output(outputs, length):
        index = tf.range(0, FLAGS.batch_size) * FLAGS.max_sequence_length + length - 1

        output_shape = _get_static_shape_as_list(outputs)
        outputs = tf.reshape(outputs, [-1, output_shape[-1]])
        return tf.gather(outputs, index)

    def _get_static_shape_as_list(tensor):
        return tensor.get_shape().as_list()

    def _get_dynamic_shape(tensor):
        return tf.shape(tensor)

    def _linear_projection(tensor, output_dim=1):
        return layers.fully_connected(tensor, output_dim, biases_initializer=None, activation_fn=None)

    """
    Model function
    --------------
    """

    @chi.model()
    def _review_model(reviews: tf.Tensor(FLAGS.batch_size, FLAGS.max_sentence_length),
                      matrix_embeddings):
        # retrieve embeddings
        # size: (batch_size x max_sentence_length x embedding_size)
        reviews = tf.nn.embedding_lookup(matrix_embeddings,
                                         reviews)
        # retrieve dynamically length of every reviews
        length = _length_sequence(reviews)

        # sum every embeddings
        sum_review = tf.reduce_sum(reviews, axis=1)
        sr_shape = tf.shape(sum_review)
        mean_review = tf.transpose(
            tf.reshape(tf.tile(length, sr_shape[1]), (sr_shape[1], sr_shape[0])))
        mean_review = tf.divide(sum_review, mean_review)
        # size: (batch_size x embedding_size)
        return mean_review

    @chi.model()
    def _product_model(reviews: tf.Tensor(FLAGS.batch_size, FLAGS.max_sentence_length, FLAGS.embedding_length),
                       ):
        cell = tf.nn.rnn_cell.LSTMCell(FLAGS.hidden_size, state_is_tuple=True)
        outputs, _ = tf.nn.dynamic_rnn(cell, reviews)
        # TODO need to retrieve all states (and not the output?)
        # size: (batch_size x max_sentence_length x hidden_size)
        return outputs

    @chi.model()
    def _user_model(reviews: tf.Tensor(FLAGS.batch_size, FLAGS.max_nb_reviews, FLAGS.embedding_size)):
        cell = tf.nn.rnn_cell.LSTMCell(FLAGS.hidden_size, state_is_tuple=True)
        outputs, _ = tf.nn.dynamic_rnn(cell, reviews, sequence_length=_length_sequence(reviews))

        output_shape = _get_static_shape_as_list(outputs)
        all_outputs = tf.reshape(outputs, (-1, output_shape[2]))
        all_outputs = layers.fully_connected(all_outputs, 1, activation_fn=tf.tanh)

        all_outputs.reshape(all_outputs, (_get_static_shape_as_list(outputs)[:-1]))
        all_outputs = tf.nn.softmax(all_outputs, dim=1)

        all_outputs = tf.tile(tf.expand_dims(all_outputs, axis=2), [1, 1, output_shape[2]])
        outputs = tf.multiply(outputs, all_outputs)
        # size: (batch_size x hidden_size)
        return tf.reduce_sum(outputs, reduction_indices=1)

    @chi.model()
    def _hop_customized_product_model(product_reviews: (FLAGS.batch_size, FLAGS.max_sentence_length, FLAGS.hidden_size),
                                      user_reviews: (FLAGS.batch_size, FLAGS.hidden_size),
                                      product_model: (FLAGS.batch_size, FLAGS.hidden_size)):
        # user_review projection
        # size: (batch_size x 1)
        user_review_projection = _linear_projection(user_reviews)

        # product_model projection
        # size: (batch_size x 1)
        product_model_projection = _linear_projection(product_model)

        # product_review projection
        # size: (batch_size * max_sequence_length x 1)
        product_review_projection = layers.linear(tf.reshape(product_reviews, (-1, product_reviews.shape[-1])), 1)
        product_review_projection = tf.reshape(product_review_projection, (
            *tf.shape(product_reviews)[:-1], 1))

        # U (formula from the paper)
        # size: (batch_size x max_sequence_length x 1)
        U = tf.tanh(product_review_projection
                    + tf.tile(tf.expand_dims(user_review_projection, axis=1),
                              [1, tf.shape(product_model_projection)[1]])
                    + tf.tile(tf.expand_dims(product_model_projection, axis=1),
                              [1, tf.shape(product_model_projection)[1]]))

        # Get a mask matrix containing 1 if an element in U is part of the sequence
        # and 0 for every element when the sequence in shorter than max_sequence_length
        # size: (batch_size x max_sequence_length x 1)
        mask = tf.reshape(
            tf.sign(
                tf.abs(
                    tf.reduce_max(product_review_projection, reduction_indices=2))),
            shape=tf.shape(U))

        nominator = tf.exp(U) * mask
        denominator = tf.reduce_sum(nominator, axis=1)

        beta = tf.div(nominator,
                      tf.tile(tf.expand_dims(denominator, axis=1), [1, tf.shape(nominator)[1]]))

        # Update the product model
        # size: (batch_size, hidden_size)
        product_model = tf.reduce_sum(
            product_reviews * tf.tile(tf.expand_dims(beta, axis=2), [1, 1, tf.shape(product_reviews)]),
            axis=1)
        return product_model

    @chi.model
    def multiple_hop(product_reviews, user_reviews, nb_hops=4):
        length = _length_sequence(product_reviews)
        # size: (batch_size, hidden_size)
        initializer = tf.reduce_sum(product_reviews, axis=1)
        initializer = tf.divide(initializer,
                                tf.tile(tf.expand_dims(length, axis=1), [1, tf.shape(product_reviews)[1]]))

        return tf.scan(lambda co_vector, _: _hop_customized_product_model(product_reviews, user_reviews, co_vector),
                       elems=tf.range(nb_hops),
                       initializer=initializer)[-1]
