import abc
import numpy as np
import mxnet as mx
from mxnet import gluon, nd, autograd

class Policy(gluon.Block, metaclass=abc.ABCMeta):
    def __init__(self, **kwargs):
        super(Policy, self).__init__(**kwargs)
        self.ce_loss = gluon.loss.SoftmaxCrossEntropyLoss()

    @abc.abstractmethod
    def forward(self, input_):
        pass

    def act(self, stochastic, input_):
        value, logits = self.forward(input_)
        if stochastic:
            action = nd.sample_multinomial(nd.softmax(logits))
        else:
            action = nd.argmax(logits, axis=-1).astype('int32')
        return action, value

    def entropy(self, logits):
        a0 = logits - nd.max(logits, axis=-1, keepdims=True)
        ea0 = nd.exp(a0)
        z0 = nd.sum(ea0, axis=-1, keepdims=True)
        p0 = ea0 / z0
        return nd.sum(p0 * (nd.log(z0) - a0), axis=-1)

    def logp(self, logits, actions):
        return -self.ce_loss(logits, actions)

    def assign_parameters(self, other_params):
        for my_par, o_par in zip(self.collect_params().values(),
                other_params.values()):
            my_par.set_data(o_par.data())


class MLPPolicy(Policy):
    def __init__(self, action_dim, layer_sizes, **kwargs):
        super(MLPPolicy, self).__init__(**kwargs)

        with self.name_scope():
            self.value = gluon.nn.Sequential()
            self.policy = gluon.nn.Sequential()
            for layer in layer_sizes:
                self.value.add(gluon.nn.Dense(layer, activation="tanh"))
                self.policy.add(gluon.nn.Dense(layer, activation="tanh"))
            self.value.add(gluon.nn.Dense(1))
            self.policy.add(gluon.nn.Dense(action_dim))

        # Seems pretty sensitive to this value
        norm_init = lambda: mx.initializer.Xavier()
        self.value.collect_params().initialize(norm_init())
        self.policy.collect_params().initialize(norm_init())

    def forward(self, input_):
        value = self.value(input_)
        logits = self.policy(input_)
        return value, logits
