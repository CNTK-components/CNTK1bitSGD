#pragma once

#include "Common.h"
#include <unordered_map>

namespace CNTK
{
    // Abstraction for learning a subset of parameters of a learnable function using first order gradient values
    class Learner
    {
    protected:
        Learner(const std::unordered_set<Variable>& parameters);

    public:
        // Method to update the parameters associated with this learner. By returning false, this method indicates that
        // learning has stopped for all of the parameters associated with this learner
        virtual bool Update(const std::unordered_map<Variable, const Value>& gradientValues, size_t trainingSampleCount) = 0;

    public:

        // Optional override that gets called before each minbatch during training
        // This is an opportunity for the learner to adapt its learning related parameters such as learning rate
        virtual void PerMinibatchCallback(const Trainer& trainer);

        // TODO: Do we need a separate callback to be called after each minibatch? 

        // Optionally overridable method to checkpoint any state associated with the learner
        virtual Dictionary Checkpoint() const;

        // Optionally overridable method to restore the learner's state from a previous checkpoint
        virtual void RestoreFromCheckpoint(const Dictionary& checkpoint);

    public:
        std::unordered_set<Variable> Parameters() const;

        DISALLOW_COPY_CTOR_AND_ASSIGNMENT(Learner);
        DISALLOW_MOVE_CTOR_AND_ASSIGNMENT(Learner);
    };

    typedef std::shared_ptr<Learner> LearnerPtr;

    // Builtin learners
    LearnerPtr SGDLearner(const std::unordered_set<Variable>& parameterVariables, double learningRatePerSample, size_t momentumTimeConstant);
    LearnerPtr AdaGradLearner(const std::unordered_set<Variable>& parameterVariables, size_t momentumTimeConstant, double gaussianNoiseInjectStd);
    LearnerPtr RmsPropLearner(const std::unordered_set<Variable>& parameterVariables, double rmsGamma);
}
