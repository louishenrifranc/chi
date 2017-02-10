import tensorflow as tf
import numpy as np
# import types
from contextlib import contextmanager
import os.path as path
import random
from .logger import logger, logging
import inspect

from tensorflow.contrib import layers
from tensorflow.contrib import framework as fw
# fw.arg_scope
from . import chi
from . import util


class SubGraph:
    """
    Basically just a wrapper around a variable scope
    with some convenience functions
    """
    stack = []

    def __init__(self, f, scope=None, default_name=None, getter=None):
        """
        Initialize a subgraph
        :param scope:
        :param default_name:
        :param getter: (relative_name) -> tf.Variable
        """
        self._reused_variables = []
        self._children = []
        self._getter = getter

        # Static variable for all SubGraph
        # If
        # it append a new Subgraph
        if SubGraph.stack:
            SubGraph.stack[-1]._children.append(self)

        with tf.variable_scope(scope, default_name, custom_getter=self._cg) as sc:
            self._scope = sc
            self.name = sc.name
            SubGraph.stack.append(self)
            self.output = f()
            SubGraph.stack.pop()

    def _cg(self, getter, name, *args, **kwargs):
        """
        Custom callback called when entering a variable scope,
        logged some information in the output
        :param getter:
        :param name:
        :param args:
        :param kwargs:
        :return:
        """
        relative_name = util.relpath(name + ':0', self._scope.name)
        v = self._getter(relative_name) if self._getter else None
        if v:
            logger.debug('reuse {}'.format(name))
            self._reused_variables.append(v)
        else:
            logger.debug('create {}'.format(name))
            v = getter(name, *args, **kwargs)

        return v

    def initialize(self):  # TODO: init from checkpoint
        """
        Init all unititiazed variables in the subgraph
        :return:
        """
        l = self.local_variables()
        g = self.global_variables()
        r = self._get_reused_variables()
        vs = l + g + r
        names = chi.get_session().run(tf.report_uninitialized_variables(vs))
        initvs = [v for v in vs if v.name[:-2].encode() in names]
        chi.get_session().run(tf.variables_initializer(initvs))

    def _get_reused_variables(self):
        vs = self._reused_variables
        for c in self._children:
            vs += c._get_reused_variables()
        return vs

    def get_ops(self):
        """
        Return all operations in the scope of the subgraph
        :return: list
            All operations in the subgraph
        """
        all_ops = tf.get_default_graph().get_operations()
        scope_ops = [x for x in all_ops if x.name.startswith(self._scope.name)]
        return scope_ops

    def get_collection(self, name):
        """
        Get the tf collection named name, in the subgraph _scope
        :param name:
        :return:
        """
        return tf.get_collection(name, self._scope.name)

    def get_ops_by_type(self, type_name):
        """
        Return Operations in the scope of the SubGraph given an operation name
        :param type_name: string (default: 'relu')
        :return:
        """
        return [op for op in self.get_ops() if op.type == type_name]

    def get_tensors_by_optype(self, type_name):
        """
        Return Tensors in the scope of the SubGraph given an operation name
        :param type_name:
        :return:
        """
        return [op.outputs[0] for op in self.get_ops_by_type(type_name)]

    def global_variables(self):
        return self.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

    def trainable_variables(self):
        return self.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

    def model_variables(self):
        return self.get_collection(tf.GraphKeys.MODEL_VARIABLES)

    def local_variables(self):
        return self.get_collection(tf.GraphKeys.LOCAL_VARIABLES)

    def losses(self):
        return self.get_collection(tf.GraphKeys.LOSSES)

    def regularization_losses(self):
        return self.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

    def summaries(self):
        """
        Return the collections of summaries
        :return:
        """
        return self.get_collection(tf.GraphKeys.SUMMARIES)

    def update_ops(self):
        return self.get_collection(tf.GraphKeys.UPDATE_OPS)
