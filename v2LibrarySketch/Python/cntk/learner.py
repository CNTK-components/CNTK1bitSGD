#
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
#

from trainer import *

class Learner:
    __metaclass__ = ABCMeta

    # Abstraction for learning a subset of parameters of a learnable function using first order gradient values
    # For e.g momentum, AdaGrad, RmsProp etc. are different types of learners with their own algorithms for 
    # learning parameter values using first order gradients.

    def __init__(self, parameters: List[Variable]):
        pass

    def update(parameterValues: Dict[Variable, Value], gradientValues: Dict[Variable, Value], trainingSampleCount: int) -> bool:
        # Method to update the parameters associated with this learner. By returning false, this method indicates that
        # learning has stopped for all of the parameters associated with this learner
        raise NotImplementedError()

    def pre_minibatch_callback(self, trainer: Trainer):
        # Optional override that gets called before each minbatch during training
        # This is an opportunity for the learner to adapt its learning related parameters such as learning rate
        pass

    # TODO: Do we need a separate callback to be called after each minibatch? 

    def get_checkpoint_state(self) -> Dictionary:
        # Optionally overridable method to get checkpoint state associated with the learner
        pass

    def restore_from_checkpoint(self, checkpoint: Dictionary):
        # Optionally overridable method to restore the learner's state from a previous checkpoint
        pass

    def parameters(self) -> List[Variable]:
        pass

# Methods to instantiate CNTK builtin learners
def sgd_learner(parameters: Set[Variable], learningRatePerSample: float, momentumTimeConstant: int, useNesterovAcceleration: bool = false) -> Learner:
    pass

def adagrad_learner(parameters: Set[Variable], momentumTimeConstant: int, gaussianNoiseInjectStd: float) -> Learner:
    pass

def rms_prop_learner(parameters: Set[Variable], rmsGamma: float) -> Learner:
    pass
