#
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
#

class TrainingControl:

    def next_minibatch_size(argument: Variable) -> int:
        # Returns the desired size of the next minibatch of the specified argument of the Model being trained
        # Note that if the value returned is different across different arguments of the model, the actual minibatch size 
        # is determined by taking a min across all arguments of the model
        pass

    def pre_minibatch_callback(trainer: Trainer) -> bool:
        # Optional callback that gets called before each minbatch during training
        # This also controls the duration of the training through its return value; false indicates end of training
        pass

    # TODO: Do we need a callback for after each minibatch too?

# Builtin training controls
def basic_training_control(maxTrainingSamplesCount: int, checkpointFrequencyinSamples: int, modelAndCheckpointFileNames: Tuple[str, str]) -> TrainingControl:
    pass

# TODO: Additional training control objects with ability of automatic minibatch size control etc.
