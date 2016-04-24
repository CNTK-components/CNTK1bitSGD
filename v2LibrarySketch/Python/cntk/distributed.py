#
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
#

from ndarray_view import *
from value import *

class DistributedWorkerDescriptor:
    global_rank # type: int
    host_id     # type: str

class DistributedCommunicator:

    # An opaque type representing the communicator object to be used for communication among distributed workers
    # Instances of this type can only be created by CNTK factory methods

    def workers(self) -> Set[DistributedWorkerDescriptor]:
        pass

    def current_worker(self) -> DistributedWorkerDescriptor:
        pass

    def sub_group(self, subGroupWorkers: Set[DistributedWorkerDescriptor]) -> DistributedCommunicator:
        # Creates a new distributed communicator comprising of a subset of the workers in this communicator
        pass

    def concatenate(self,
                    inValues: Set[Value],
                    sendToWorkers: Set[DistributedWorkerDescriptor],
                    concatenatedValues: Set[Value],
                    device: DeviceDescriptor = DeviceDescriptor.default_device()):
        # A collective communication API to concatenate values across each worker of this communicator. The concatenated values are only sent to the specified workers; for all others the returned Values are null
        # TODO: Add an async variant of the Concatenate method
        pass

    def aggregate(self,
                  inValues: Set[Value],
                  sendToWorkers: Set[DistributedWorkerDescriptor], 
                  aggregatedOutputs: Set[Value],
                  device: DeviceDescriptor = DeviceDescriptor.default_device()):
        # A collective communication API to aggregate values across each worker of this communicator. The agrregated values are only sent to the specified workers; for all others the returned Values are null
        # TODO: Add an async variant of the Aggregate method
        pass

    def quantized_aggregate(self,
                            inValues: Set[Value],
                            inPreviousQuantizationResidues: Set[Value],
                            sendToWorkers: Set[DistributedWorkerDescriptor],
                            aggregatedOutputs: Set[Value],
                            newQuantizationResidues: Set[Value],
                            device: DeviceDescriptor = DeviceDescriptor.default_device()):
        # A collective communication API to perform quantized aggregation of values across all workers of this communicator
        # TODO: Add an async variant of the QuantizedAggregate method
        pass

class DistributedTrain:
    def __init__(self, communicator: DistributedCommunicator):
        pass

    def pre_parameter_update_callback(self, trainer: Trainer, gradientValues: Dict[Variable, Value]):
        # Optional override that gets called per minibatch after finishing gradient computation but before updating model parameters
        pass

    def pre_minibatch_callback(self, trainer: Trainer):
        # Optional override that gets called before each minbatch during training
        pass

    def get_checkpoint_state(self) -> Dictionary:
        # Optionally overridable method to get checkpoint state associated with this Distributed train method
        pass

    def restore_from_checkpoint(self, checkpoint: Dictionary):
        # Optionally overridable method to restore state pertaining this distributed training method from a previous checkpoint
        pass

    def communicator(self) -> DistributedCommunicator:
        pass

def mpi_communicator() -> DistributedCommunicator:
    # Built-in communicators
    pass

# Builtin distributed training methods
# TODO: Model parallelism

def data_parallel(communicator: DistributedCommunicator, numGradientQuantizationLevels: int, useAsyncBufferedParameterUpdate: bool = false) -> DistributedTrain:
    # Per minibatch synchronous data-parallel training that aggregates gradients computed across all workers
    pass

def model_averaging(communicator: DistributedCommunicator, averagingFrequency: int) -> DistributedTrain:
    # Model Averaging; 
    pass

# TODO: Add Block-momentum support
