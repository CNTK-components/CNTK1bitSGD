#pragma once

#include "Common.h"
#include <unordered_set>
#include "Trainer.h"

namespace CNTK
{
    struct DistributedWorkerDescriptor
    {
        size_t m_globalRank;
        std::wstring m_hostId;
    };

    // An opaque type representing the communicator object to be used for communication among distributed workers
    // Instances of this type can only be created by CNTK factory methods
    class DistributedCommunicator
    {
    public:
        std::unordered_set<DistributedWorkerDescriptor> Workers() const;

        // Creates a new distributed communicator comprising of a subset of the workers in this communicator
        DistributedCommunicator SubGroup(const std::unordered_set<DistributedWorkerDescriptor>& subGroupWorkers) const;

        // A collective communication API to concatenate values across each worker of this communicator. The concatenated values are only sent to the specified workers; for all others the returned Values are null
        std::unordered_set<Value> Concatenate(const std::unordered_set<Value>& values, const std::unordered_set<DistributedWorkerDescriptor>& sendToWorkers, DeviceDescriptor device = DeviceDescriptor::DefaultDevice());

        // A collective communication API to aggregate values across each worker of this communicator. The agrregated values are only sent to the specified workers; for all others the returned Values are null
        void Aggregate(const std::unordered_set<Value>& inValues,
                       const std::unordered_set<DistributedWorkerDescriptor>& sendToWorkers, 
                       const std::unordered_map<Variable, Value>& aggregatedOutputs,
                       DeviceDescriptor device = DeviceDescriptor::DefaultDevice());

        // A collective communication API to perform quantized aggregation of values across all workers of this communicator
        void QuantizedAggregate(const std::unordered_set<Value>& inValues,
                                const std::unordered_set<Value>& inPreviousQuantizationResidues,
                                const std::unordered_set<DistributedWorkerDescriptor>& sendToWorkers,
                                const std::unordered_map<Variable, Value>& aggregatedOutputs,
                                const std::unordered_map<Variable, Value>& newQuantizationResidues,
                                DeviceDescriptor device = DeviceDescriptor::DefaultDevice());

        DistributedCommunicator() = delete;
    };

    class DistributedTrain
    {
    protected:
        DistributedTrain(DistributedCommunicator communicator);

    public:

        // Optional override that gets called per minibatch after finishing gradient computation but before updating model parameters
        virtual void PreParameterUpdateCallback(const Trainer& trainer, const std::unordered_map<Variable, const Value>& gradientValues);

        // Optional override that gets called before each minbatch during training
        virtual void PerMinibatchCallback(const Trainer& trainer);

        // Optionally overridable method to checkpoint any state associated with this Distributed train method
        virtual Dictionary Checkpoint() const;

        // Optionally overridable method to restore state pertaining this distributed training method from a previous checkpoint
        virtual void RestoreFromCheckpoint(const Dictionary& checkpoint);

    public:
        DistributedCommunicator Communicator() const;

        DistributedTrain() = delete;

        DISALLOW_COPY_CTOR_AND_ASSIGNMENT(DistributedTrain);
        DISALLOW_MOVE_CTOR_AND_ASSIGNMENT(DistributedTrain);
    };

    typedef std::shared_ptr<DistributedTrain> DistributedTrainPtr;

    // Built-in communicators
    DistributedCommunicator MPICommunicator();

    // Builtin distributed training methods
    DistributedTrainPtr DataParallel(DistributedCommunicator communicator, size_t numGradientQuantizationLevels, bool useAsyncBufferedParameterUpdate = false);
    DistributedTrainPtr ModelParallel(DistributedCommunicator communicator, size_t averagingFrequency);
}
