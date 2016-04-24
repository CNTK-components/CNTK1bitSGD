#
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
#

from learner import *
from function import *

class Trainer:

    def __init__(self, model: Function, trainingLoss: Variable, modelParameterLearners: Set[Learner], outputs: Set[Variable] = None, distributedTrain: DistributedTrain = None):
        pass

    def train_minibatch(self, arguments: Dict[Variable, Value]) -> bool:
        # Optimizes the model parameters using the specified "arguments" values for model Arguments
        # The return value is false if all model parameter learners indicate end of learning (through their Update method's return value)
        pass

    def train(self, reader: MinibatchSource, modelArgumentsToMinibatchSourceStreamMap: Dict[Variable, StreamDescription], controller: TrainingControl):
        # Trains the model with data continuously fed by the specified 'reader' and duration of  training determined by the specified TrainingControl object
        # The 'modelArgumentsToMinibatchSourceStreamMap' argument specifies a 1-1 mapping between the model's argument variables and the reader stream that corresponds to that argument
        pass

    def number_of_training_samples_processed(self) -> int:
        pass

    def model(self) -> Function:
        pass

    def training_loss_variable(self) -> Variable:
        pass

    def last_minibatch_training_loss(self) -> float:
        pass

    def last_minibatch_outputs(self) -> Dict[Variable, Value]:
        pass

    def learners(self) -> Set[Learner]:
        pass

    def distributed_train(self) -> DistributedTrain:
        pass

    def write_checkpoint(self, modelFilePath: str, checkpointFilePath: str):
        pass

    def restore_from_checkpoint(self, modelFilePath: str, checkpointFilePath: str):
        pass
