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
        ValueType m_elementType;       // Element type of the stream
        NDShape m_sampleLayout;        // Layout of the sample for the stream
    };

    class Reader
    {
    public:
        // Describes the streams this reader produces.
        virtual std::unordered_set<StreamDescription> GetStreamDescriptions() = 0;

        // Reads a minibatch that contains data across all streams.
        // The minibatchData argument specifies the desired minibatch size for each stream of the reader and the actual returned size is the min across all streams
        // The return value of false indciates that the reader will no longer return any further data
        virtual bool GetNextMinibatch(std::unordered_map<StreamDescription, std::pair<size_t, Value>>& minibatchData) = 0;

        // Positions the reader stream to the specified position on the global timeline
        virtual void ResetPosition(size_t newPosition);

        // TODO: Methods to save and restore from checkpoints

        DISALLOW_COPY_CTOR_AND_ASSIGNMENT(Reader);
        DISALLOW_MOVE_CTOR_AND_ASSIGNMENT(Reader);
    };

    typedef std::shared_ptr<Reader> ReaderPtr;

    // Helper method to get the stream description for the first stream matching the specified 
    StreamDescription GetStreamDescription(ReaderPtr reader, std::wstring streamName);

    // Builtin readers
    ReaderPtr TextReader(/*Text Reader configuration parameters*/);
}
