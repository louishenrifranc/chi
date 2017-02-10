import tensorflow as tf
from .model import Model
from . import util
import chi
import inspect
import numpy as np
from chi.logger import logger
import os


def function(f=None, logdir=None, *args, **kwargs):
    """
    Decorator
    :param f:
    :param logdir:
    :param args:
    :param kwargs:
    :return:
    """
    if f:  # use as function
        if isinstance(f, Function):
            return f
        else:
            return Function(f, logdir, *args, **kwargs)
    else:  # use with @ as decorator
        return lambda f: Function(f, logdir, *args, **kwargs)


class Function(Model):
    def __init__(self, f, logdir=None, *args, **kwargs):
        """

        :param f: function
        :param logdir: string
        :param args:
        :param kwargs:
        """
        self._step = 0
        # Init model parent class
        Model.__init__(self, f, *args, **kwargs)

        # process inputs
        import collections

        # for each parameters, inputs map parameter name to the placeholder
        self.inputs = collections.OrderedDict()

        # for each parameter, auto_wrap map parameter name to a boolean if the shape is a list
        self.auto_wrap = collections.OrderedDict()
        for name, dtype, shape, default in parse_signature(f):
            # If a default value has been provided, then we don't need to provide an output for this placeholder
            # in the graph, default will be the output
            if default:
                p = tf.placeholder_with_default(default, shape)
            else:
                p = tf.placeholder(dtype, shape, name)
            self.auto_wrap.update({name: isinstance(shape, list)})
            self.inputs.update({name: p})

        # if any parameter has a shape
        self.use_wrap = any(self.auto_wrap.values())

        # build graph
        out = super().__call__(**self.inputs)  # build Model
        self.__dict__.update(self._last_graph.__dict__)  # make SubGraph properties available in self

        # process outputs
        if out is None:
            self.output = tf.no_op()
        # elif self.use_wrap:
        #   self.unwrap = []
        #   self.output = []
        #   if isinstance(out, tuple):
        #     for x in out:
        #       unwrap = isinstance(x, list)
        #       if unwrap:
        #         assert len(x) == 1 and isinstance(x[0], tf.Tensor)
        #         x = x[0]
        #       self.unwrap.append(unwrap)
        #       self.output.append(x)
        #   elif isinstance(out, list):
        #     assert len(out) == 1 and isinstance(out[0], tf.Tensor)
        #     self.output = out[0]

        # self.inputs = self.get_tensors_by_optype("Placeholder")

        # If a log dir is specified, then create a FileWriter, passing the graph just defined
        if logdir:
            current_app = chi.App.current_app
            if not logdir.startswith('/'):
                logger.debug('logdir path relative to app: {}, app logdir: {}'.format(current_app, current_app.logdir))
                if current_app and current_app.logdir:
                    logdir = current_app.logdir + '/' + logdir
                else:
                    logger.debug('fall back to logdir path relative to working dir')
                    os.path.abspath('./' + logdir)
            # Write the session graph
            self.writer = tf.summary.FileWriter(logdir, graph=chi.chi.get_session().graph)
        else:
            self.writer = None

        # collect activations tensor
        activations = self.get_tensors_by_optype('Relu')  # TODO: generalize to non-Relu
        # activations = self.subgraph.histogram_summaries(activations, 'activations')
        summaries = self.summaries()

        # If there are summaries, and a writer to write to it
        if summaries and self.writer:
            self._summary_op = tf.summary.merge(summaries)

        super().initialize()

    def __call__(self, *args, **kwargs):
        """
        Called when the function decorated is actually called
        :param args:
        :param kwargs:
        :return:
        """
        # Map Tensor -> input value
        feeds = {}
        use_wrap = False
        for p, auto_wrap, arg in zip(self.inputs.values(), self.auto_wrap.values(), args):
            if auto_wrap:
                arg = np.array(arg)
                if p.get_shape().is_compatible_with(arg.shape):
                    assert not use_wrap
                else:
                    arg = arg[np.newaxis, ...]  # add batch dimension
                    use_wrap = True
                    assert p.get_shape().is_compatible_with(arg.shape)

            feeds[p] = arg
        feeds.update(kwargs)  # TODO: process kwargs correctly

        # Run the session
        results = self.run_log(self.output, feeds)
        if use_wrap:
            # remove batch dimension
            if isinstance(results, (tuple, list)):
                results = [r[0, ...] if r.shape[0] == 1 else r for r in results if isinstance(r, np.ndarray)]
            else:
                results = results[0, ...] if isinstance(results, np.ndarray) and results.shape[0] == 1 else results

        self._step += 1
        return results

    def run_log(self, fetches, feeds):
        """
        Run the session
        :param fetches: Operation to run in the subgraph
        :param feeds: Feed dictionnary
        :return: Values from the Operation
        """
        log = self.writer  # TODO: good default logging policy
        if log:
            # TODO(check) Run twice summary_op, remove it...
            results, summary = chi.chi.get_session().run((fetches, self._summary_op), feeds)
            self.writer.add_summary(summary, global_step=self._step)
        else:
            results = chi.chi.get_session().run(fetches, feeds)
        return results

    def reset(self):
        local_inits = [v.initializer for v in self.local_variables()]
        chi.chi.get_session().run(local_inits)

    # TODO:
    def save(self):
        # save parameters etc.
        # if (self.t+45000) % 50000 == 0: # TODO: correct
        #   s = self.saver.save(self.sess,FLAGS.outdir+"f/tf/c",self.t)
        #   print("DDPG Checkpoint: " + s)
        pass

        # TODO: SubGraph as class method
        # def __get__(self, obj, objtype):
        #   """
        #   In case the SubGraph is a class method we need to instantiate it for every instance.
        #   By implementing __get__ we make it a property which allows us to instantiate a new SubGraph
        #   the first time it is used.
        #   """
        #   if obj:
        #     setattr(obj, self.f.__name__, SubGraph(self.f, parent=obj))
        #   else:
        #     # if we are called directly from the class (not the instance) TODO: error?
        #     pass


def parse_signature(f):
    """
    Parse the signature of the function.
    Return a list of every parameters
    Each parameter is of form [name, dtype, shape, default_value]
    :param f:
    :return:
    """
    sig = inspect.signature(f)
    # out is a list containing (name of the tensor, type, shape, default_value if passed, else None
    out = []
    # iterate over all parameters
    for n, v in sig.parameters.items():
        if not v.default == inspect.Parameter.empty:
            t = tf.convert_to_tensor(v.default)
            out.append((n, t.dtype, t.get_shape(), v.default))
        elif not v.annotation == inspect.Parameter.empty:
            out.append((n, *util.type_and_shape_from_annotation(v.annotation), None))
        else:
            out.append((n, tf.float32, None, None))
    return out


def test_runnable():
    @function
    def f(a, b):
        return a * b

    assert f(3, 2) == 6
