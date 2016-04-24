#pragma once

#include "Common.h"
#include "Function.h"

namespace CNTK
{
    // This class describes a particular stream: its name, element type, storage, etc.
    struct StreamDescription
    {
        std::wstring m_name;           // Unique name of the stream
        size_t m_id;                   // Unique identifier of the stream
        StorageType m_storageType;     // Storage type of the stream
        DataType m_elementType;       // Element type of the stream
        NDShape m_sampleLayout;        // Layout of the sample for the stream
    };

    class MinibatchSource
    {
    public:
        // Describes the streams this reader produces.
        virtual std::unordered_set<StreamDescription> GetStreamDescriptions() = 0;

        // Reads a minibatch that contains data across all streams.
        // The minibatchData argument specifies the desired minibatch size for each stream of the reader and the actual returned size is the min across all streams
        // The return value of false indciates that the reader will no longer return any further data
        // TODO: Distributed reading support
        virtual bool GetNextMinibatch(std::unordered_map<StreamDescription, std::pair<size_t, Value>>& minibatchData) = 0;

        // Positions the reader stream to the specified position on the global timeline
        virtual void ResetPosition(size_t newPosition);

        // TODO: Methods to save and restore from checkpoints
    };

    typedef std::shared_ptr<MinibatchSource> MinibatchSourcePtr;

    // Helper method to get the stream description for the first stream matching the specified 
    StreamDescription GetStreamDescription(MinibatchSourcePtr reader, std::wstring streamName);

    // Methods to instantiate CNTK built-in MinibatchSources 
    MinibatchSourcePtr TextMinibatchSource(/*Text MinibatchSource configuration parameters*/);
}
