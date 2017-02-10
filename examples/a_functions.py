""" demo of chi.function
This is how we can use python functions with the
chi.function decorator to build and execute TensorFlow graphs.
"""
import tensorflow as tf
import chi
import numpy as np


@chi.function
def my_tf_fun(x, y):
    z = tf.nn.relu(x) * y
    return z


my_tf_fun(3, 5)


# we can also specify shapes (using python3's annotation syntax)
@chi.function
def my_tf_fun(x: (2, 3), y):
    z = tf.nn.relu(x) * y
    return z


my_tf_fun(np.ones((2, 3)), -8)


# the first dimension is often the batch dimension and is required
# for the Keras-style tf.contrib.layers
# With a special syntax, chi.function can automatically add that dimension and remove it
# from the result if it is == 1
@chi.function
def my_tf_fun(x: [[3]], y):  # [shape] activates auto wrap
    z = tf.nn.relu(x) * y
    return z


assert np.all(my_tf_fun(np.zeros([32, 3]), 5) == np.zeros([32, 3]))  # with batch dimension
assert np.all(my_tf_fun(np.zeros([3]), 5) == np.zeros([3]))  # without batch dimension


# chi.function also helps with logging see c_experiments.py for that


# Btw.: these @ decorators are just a shortcut for:
def my_tf_fun(x, y):
    z = tf.nn.relu(x) * y
    return z


my_tf_fun = chi.function(my_tf_fun)

assert my_tf_fun(3, 5) == 15.

# other than decorators, this does not break type inference and auto complete
