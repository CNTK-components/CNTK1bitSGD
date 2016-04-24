#pragma once

#include "Common.h"
#include <unordered_set>
#include "Variable.h"
#include "Value.h"

namespace CNTK
{
    class TrainingControl
    {
    public:
        // Returns the desired size of the next minibatch of the specified argument of the Model being trained
        // Note that if the value returned is different across different arguments of the model, the actual minibatch size 
        // is determined by taking a min across all arguments of the model
        virtual size_t NextMinibatchSize(Variable argument) = 0;

    public:
        // Optional callback that gets called before each minbatch during training
        // This also controls the duration of the training through its return value; false indicates end of training
        virtual bool PreMinibatchCallback(const Trainer& trainer);

        // TODO: Do we need a callback for after each minibatch too?
    };

    typedef std::shared_ptr<TrainingControl> TrainingControlPtr;

    // Builtin training controls
    TrainingControlPtr BasicTrainingControl(size_t maxTrainingSamplesCount, size_t checkpointFrequencyinSamples, const std::pair<std::wstring, std::wstring>& modelAndCheckpointFileNames);

    // TODO: Additional training control objects with ability of automatic minibatch size control etc.
}
