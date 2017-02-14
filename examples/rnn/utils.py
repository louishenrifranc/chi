import tensorflow as tf
from tensorflow.contrib import layers
import functools


def _length_sequence(sentences):
    used = tf.sign(tf.reduce_sum(tf.abs(sentences), reduction_indices=2))
    length = tf.reduce_sum(used, reduction_indices=1)
    return tf.cast(length, tf.int32)


def _get_static_shape_as_list(tensor):
    return tensor.get_shape().as_list()


def _get_dynamic_shape(tensor):
    return tf.shape(tensor)


def linear_projection(tensor, output_dim=1):
    return layers.fully_connected(tensor, output_dim, biases_initializer=None, activation_fn=None)


def doublewrap(function):
    """
    A decorator decorator, allowing to use the decorator to be used without
    parentheses if not arguments are provided. All arguments must be optional.
    """

    @functools.wraps(function)
    def decorator(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            return function(args[0])
        else:
            return lambda wrapee: function(wrapee, *args, **kwargs)

    return decorator


@doublewrap
def define_scope(function, scope=None, *args, **kwargs):
    """
    A decorator for functions that define TensorFlow operations. The wrapped
    function will only be executed once. Subsequent calls to it will directly
    return the result so that operations are added to the graph only once.
    The operations added by the function live within a tf.variable_scope(). If
    this decorator is used with arguments, they will be forwarded to the
    variable scope. The scope name defaults to the name of the wrapped
    function.
    """
    attribute = '_cache_' + function.__name__
    name = scope or function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            with tf.variable_scope(name, *args, **kwargs):
                setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator
