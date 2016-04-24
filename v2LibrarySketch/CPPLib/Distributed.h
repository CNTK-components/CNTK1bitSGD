#pragma once

#include "Common.h"
#include <unordered_set>
#include "Trainer.h"

namespace CNTK
{
    struct DistributedWorkerDescriptor
    {
        size_t GlobalRank;
        std::wstring HostId;
    };

    // An opaque type representing the communicator object to be used for communication among distributed workers
    // Instances of this type can only be created by CNTK factory methods
    class DistributedCommunicator
    {
    public:
        std::unordered_set<DistributedWorkerDescriptor> Workers() const;
        DistributedWorkerDescriptor CurrentWorker();

        // Creates a new distributed communicator comprising of a subset of the workers in this communicator
        DistributedCommunicator SubGroup(const std::unordered_set<DistributedWorkerDescriptor>& subGroupWorkers) const;

        // A collective communication API to concatenate values across each worker of this communicator. The concatenated values are only sent to the specified workers; for all others the returned Values are null
        // TODO: Add an async variant of the Concatenate method
        std::unordered_set<Value> Concatenate(const std::unordered_set<Value>& values, const std::unordered_set<DistributedWorkerDescriptor>& sendToWorkers, DeviceDescriptor device = DeviceDescriptor::DefaultDevice());

        // A collective communication API to aggregate values across each worker of this communicator. The agrregated values are only sent to the specified workers; for all others the returned Values are null
        // TODO: Add an async variant of the Aggregate method
        void Aggregate(const std::unordered_set<Value>& inValues,
                       const std::unordered_set<DistributedWorkerDescriptor>& sendToWorkers, 
                       const std::unordered_set<Value>& aggregatedOutputs,
                       DeviceDescriptor device = DeviceDescriptor::DefaultDevice());

        // A collective communication API to perform quantized aggregation of values across all workers of this communicator
        // TODO: Add an async variant of the QuantizedAggregate method
        void QuantizedAggregate(const std::unordered_set<Value>& inValues,
                                const std::unordered_set<Value>& inPreviousQuantizationResidues,
                                const std::unordered_set<DistributedWorkerDescriptor>& sendToWorkers,
                                const std::unordered_set<Value>& aggregatedOutputs,
                                const std::unordered_set<Value>& newQuantizationResidues,
                                DeviceDescriptor device = DeviceDescriptor::DefaultDevice());

        DistributedCommunicator() = delete;
    };

    class DistributedTrain
    {
    protected:
        DistributedTrain(DistributedCommunicator communicator);

    public:

        // Optional override that gets called per minibatch after finishing gradient computation but before updating model parameters
        virtual void PreParameterUpdateCallback(const Trainer& trainer, const std::unordered_map<Variable, Value>& gradientValues);

        // Optional override that gets called before each minbatch during training
        virtual void PreMinibatchCallback(const Trainer& trainer);

        // Optionally overridable method to get checkpoint state associated with this Distributed train method
        virtual Dictionary GetCheckpointState() const;

        // Optionally overridable method to restore state pertaining this distributed training method from a previous checkpoint
        virtual void RestoreFromCheckpoint(const Dictionary& checkpoint);

    public:
        DistributedCommunicator Communicator() const;

        DistributedTrain() = delete;
    };

    typedef std::shared_ptr<DistributedTrain> DistributedTrainPtr;

    // Built-in communicators
    DistributedCommunicator MPICommunicator();

    // Builtin distributed training methods

    // Per minibatch synchronous data-parallel training that aggregates gradients computed across all workers
    DistributedTrainPtr DataParallel(DistributedCommunicator communicator, size_t numGradientQuantizationLevels, bool useAsyncBufferedParameterUpdate = false);

    // Model Averaging; TODO: Add Block-momentum support
    DistributedTrainPtr ModelAveraging(DistributedCommunicator communicator, size_t averagingFrequency);

    // TODO: Model parallelism
}
