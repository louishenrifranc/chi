from tensorflow.contrib import layers
from utils import linear_projection, define_scope
import argparse
import tensorflow as tf
import numpy as np

print(tf.__version__)
class Model:
    def __init__(self, args):
        self.max_sentence_length = args.max_sentence_length  # maximum length for a sentence
        self.embedding_words_dim = args.embedding_words_dim  # size of the word embeddings
        self.hidden_size = args.hidden_size  # nb hidden layer
        self.max_nb_reviews = args.max_nb_reviews  # maximum number of reviews for a product or a user in one batch example

        self.user_reviews = tf.placeholder(shape=(None, self.max_sentence_length),
                                           dtype=tf.int32)
        self.product_reviews = tf.placeholder(shape=(None, self.max_sentence_length),
                                              dtype=tf.int32)

        self.product_reviews_score = tf.placeholder(shape=None, dtype=tf.float32)
        self.target_score = tf.placeholder(shape=(), dtype=tf.int32)

        self.optimizer = tf.train.AdadeltaOptimizer()

        if args.glove_embeddings is not None:
            self.word_embeddings = tf.Variable(initial_value=args.glove_embeddings, name="word_embedding")
        else:
            self.word_embeddings = tf.Variable(
                np.random.standard_normal(size=(args.vocab_size, self.embedding_words_dim)),
                name="word_embedding")

        self.optimize()

    def prediction(self):
        """
        Main function, do the forward propagation, joining blocks together
        :return:
        """
        with tf.variable_scope("prediction"):
            self.nn_lookup()
            self._compute_mask()

            self.product_reviews = self._product_model()

            self.user_reviews = self._to_sentence_embedding(self.user_reviews, self.nb_reviews_users)

            self.product_reviews = self._to_sentence_embedding(self.product_reviews, self.nb_reviews_product)
            self.product_reviews = self._user_model()

            self.hop_pass(4)
            return self.opininon_rating()

    def nn_lookup(self):
        with tf.variable_scope("nn_lookup"):
            self.product_reviews = tf.nn.embedding_lookup(self.word_embeddings, self.product_reviews)
            self.user_reviews = tf.nn.embedding_lookup(self.word_embeddings, self.user_reviews)

    def optimize(self):
        with tf.variable_scope("optimize"):
            opinion_rating = self.prediction()
            loss = tf.square(opinion_rating - self.target_score)
            self.opt = self.optimizer.minimize(loss)

    def _compute_mask(self):
        """
        Compute sentence length for all reviews (useful afterward for padding)
        """

        def nb_reviews(matrix):
            result = tf.sign(tf.reduce_sum(tf.abs(matrix), axis=2))
            return tf.reduce_sum(result, axis=1)

        with tf.variable_scope("compute_mask"):
            # Number of reviews for each user
            # Size: nb_reviews_user
            self.nb_reviews_users = nb_reviews(self.user_reviews)

            # Number of reviews per each product
            # Size: nb_reviews_product
            self.nb_reviews_product = nb_reviews(self.product_reviews)

    def _to_sentence_embedding(self, reviews, reviews_length):
        """
        Compute the average mean of word embedding in every review of a reviewer
        :return:
            Sentence embedding for every reviews
        """
        with tf.variable_scope("to_sentence_embedding"):
            # Size: nb_reviews_user x word_embedding_size
            mean_review = tf.reduce_sum(reviews, axis=1)

            mean_review_shape = tf.shape(mean_review)

            # Divide the sum of word embedding in a review by the number of word in a review
            # Size: nb_reviews_user x word_embedding_size
            mean_review /= tf.transpose(tf.reshape(tf.tile(reviews_length, mean_review_shape[1]),
                                                   (mean_review_shape[1], mean_review_shape[0])))
            return mean_review

    def _product_model(self):
        """
        Compute the embedding for the product reviews
        :return:
            Word embedding for product embedding
        """
        with tf.variable_scope("product_model"):
            # Size: 1 x nb_product_reviews x word_embedding_size
            self.product_reviews = tf.reshape(self.product_reviews, (1, -1, self.embedding_words_dim))
            cell = tf.contrib.rnn.LSTMCell(self.hidden_size)
            outputs, _ = tf.nn.dynamic_rnn(cell, self.product_reviews)

            # TODO it's actually wrong, i just don't know how to return every hidden state step, so i return every output step
            # Size: nb_product_reviews x hidden_size
            return outputs

    def _user_model(self):
        """
        Compute a single vector representing a user
        It maps every reviews from the user in a recurrent neural network, retrieve the hidden (TODO right now its the output)
        and compute an average sum
        :return:
            A single vector representing the user
        """
        with tf.variable_scope("user_model"):
            cell = tf.contrib.rnn.LSTMCell(self.hidden_size)

            self.user_reviews = tf.reshape(self.user_reviews, (1, -1, self.embedding_words_dim))
            outputs, _ = tf.nn.dynamic_rnn(cell, self.user_reviews)

            outputs = tf.unstack(outputs)
            assert len(tf.shape(outputs)) == 2, "must check the size of the output of a rnn in tf"

            weighted_sum = layers.fully_connected(outputs, 1, activation_fn=tf.tanh)
            weighted_sum = tf.nn.softmax(weighted_sum, dim=0)
            weighted_sum = tf.tile(tf.expand_dims(weighted_sum, axis=1), [1, self.hidden_size])

            average_sum = weighted_sum * outputs
            # Size: Hidden_size
            return tf.reduce_sum(average_sum, reduction_indices=1)

    def _hop_customized_product_model(self, co_vector):
        with tf.variable_scope("hop_customized_product_model"):
            # Size: scalar
            user_review_projection = tf.unstack(linear_projection(self.user_reviews))

            # Size: scalar
            co_vector = tf.unstack(linear_projection(co_vector))

            # Size: nb_product_reviews
            product_review_projection = layers.linear(self.product_reviews, 1)

            score = tf.tanh(user_review_projection + co_vector + user_review_projection)
            beta = tf.nn.softmax(score)

            beta = tf.tile(tf.expand_dims(beta, axis=1), [1, self.hidden_size])
            return self.product_reviews * beta

    def hop_pass(self, nb_hops):
        with tf.variable_scope("hop_pass"):
            initializer = tf.reduce_mean(self.product_reviews, axis=0)

            product_vector = tf.scan(lambda vector, _: self._hop_customized_product_model(vector),
                                     elems=tf.range(nb_hops),
                                     initializer=initializer[-1])
            return product_vector

    def opininon_rating(self):
        with tf.variable_scope("opinion_rating"):
            self.mu = tf.get_variable(name="mu", shape=1)
            output_rating = self.mu * layers.fully_connected(self.product_reviews, 1, activation_fn=tf.tanh)
            return output_rating + tf.reduce_mean(self.target_score)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_sentence_length', type=int, default=100, help='Size of a batch')
    parser.add_argument('--embedding_words_dim', type=int, default=28, help='Size of a batch')
    parser.add_argument('--hidden_size', type=int, default=32, help='Size of a batch')
    parser.add_argument('--max_nb_reviews', type=int, default=32, help='Size of a batch')
    parser.add_argument('--glove_embeddings', type=int, default=None, help='Size of a batch')
    parser.add_argument('--vocab_size', type=int, default=100, help='Size of a batch')

    args, _ = parser.parse_known_args()

    user_reviews = np.random.randint(0, args.vocab_size,
                                     size=(np.random.randint(0, args.max_nb_reviews), args.max_sentence_length))
    for row in range(user_reviews.shape[0]):
        stop_sentence = np.random.randint(0, args.max_sentence_length)
        user_reviews[row, stop_sentence:] = 0

    prod_review = np.random.randint(0, args.vocab_size,
                                    size=(np.random.randint(0, args.max_nb_reviews), args.max_sentence_length))
    for row in range(prod_review.shape[0]):
        stop_sentence = np.random.randint(0, args.max_sentence_length)
        prod_review[row, stop_sentence:] = 0

    prod_ratings = np.random.randint(0, 10, size=prod_review.shape[0])
    user_rating = np.random.randint(0, 10)

    model = Model(args)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(model.opt, feed_dict={model.user_reviews: user_reviews,
                                       model.product_reviews: prod_review,
                                       model.product_reviews_score: prod_ratings,
                                       model.target_score: user_rating})
