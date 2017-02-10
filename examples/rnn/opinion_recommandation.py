import tensorflow_fold as td
from tensorflow.contrib import layers

import tensorflow as tf
import numpy as np
import chi

FLAGS = tf.app.flags.FLAGS
FLAGS.DEFINE_integer('embedding_size', 32, 'embedding size of word in reviews')
FLAGS.DEFINE_integer('batch_size', 64, 'batch size')
FLAGS.DEFINE_integer('max_sentence_length', 128, 'max sentence length')


@chi.experiment(local_dependencies=[chi])
def model():
    @chi.function()
    def _word_embeddings_to_sentence(reviews: tf.Tensor(FLAGS.batch_size, FLAGS.max_sentence_length),
                                     matrix_embeddings):
        reviews = tf.nn.embedding_lookup(matrix_embeddings,
                                         reviews)  # batch_size x max_sentence_length x embedding_size
        # get length of every reviews
        length = _length_sentences(reviews)

        sum_review = tf.reduce_sum(reviews, axis=1)
        mean_review = tf.transpose(
            tf.reshape(tf.tile(length, sum_review.shape[1]), (sum_review.shape[1], sum_review.shape[0])))
        mean_review = tf.divide(sum_review, mean_review)
        return mean_review  # batch_size x embedding_size

    @chi.function()
    def _length_sentences(sentences):
        used = tf.sign(tf.reduce_sum(tf.abs(sentences), reduction_indices=2))
        length = tf.reduce_sum(used, reduction_indices=1)
        return tf.cast(length, tf.int32)

    @chi.function()
    def _get_state_output(sentences):
        pass

    @chi.function()
    def _repeat_elements(x, repeats, axis):
        '''Repeats the elements of a tensor along an axis, like np.repeat
        If x has shape (s1, s2, s3) and axis=1, the output
        will have shape (s1, s2 * rep, s3)
        This function is taken from keras backend
        '''
        x_shape = x.get_shape().as_list()
        splits = tf.split(axis, x_shape[axis], x)
        x_rep = [s for s in splits for _ in range(rep)]
        return tf.concat(axis, x_rep)

    @chi.function()
    def _product_reviews(reviews: tf.Tensor(FLAGS.batch_size, FLAGS.max_sentence_length, FLAGS.embedding_length),
                         ):
        cell = tf.nn.rnn_cell.LSTMCell(FLAGS.hidden_size, state_is_tuple=True)
        outputs, _ = tf.nn.dynamic_rnn(cell, reviews)
        # need to retrieve all states
        return outputs  # batch_size x max_sentence_length x hidden_size

    @chi.function()
    def _user_reviews(reviews: tf.Tensor(FLAGS.batch_size, FLAGS.max_nb_reviews, FLAGS.embedding_size)):
        cell = tf.nn.rnn_cell.LSTMCell(FLAGS.hidden_size, state_is_tuple=True)
        outputs, _ = tf.nn.dynamic_rnn(cell, reviews)

        all_outputs = tf.reshape(outputs, (outputs.shape[0] * outputs.shape[1], outputs.shape[2]))
        all_outputs = layers.fully_connected(all_outputs, 1, activation_fn=tf.tanh)

        all_outputs.reshape(all_outputs, (outputs.shape[:-1]))
        all_outputs = tf.nn.softmax(all_outputs, dim=1)
        all_outputs = _repeat_elements(all_outputs, repeats=outputs.shape[2], axis=2)
        outputs = tf.multiply(outputs, all_outputs)
        return tf.sum(outputs, axis=1)  # batch_size x hidden_size
