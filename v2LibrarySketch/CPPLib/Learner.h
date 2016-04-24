#pragma once

#include "Common.h"
#include <unordered_map>

namespace CNTK
{
    // Abstraction for learning a subset of parameters of a learnable function using first order gradient values
    // For e.g momentum, AdaGrad, RmsProp etc. are different types of learners with their own algorithms for 
    // learning parameter values using first order gradients.
    class Learner
    {
    protected:
        Learner(const std::vector<Variable>& parameters);

    public:
        // Method to update the parameters associated with this learner. By returning false, this method indicates that
        // learning has stopped for all of the parameters associated with this learner
        virtual bool Update(const std::unordered_map<Variable, Value>& parameterValues,
                            const std::unordered_map<Variable, Value>& gradientValues,
                            size_t trainingSampleCount) = 0;

    public:

        // Optional override that gets called before each minbatch during training
        // This is an opportunity for the learner to adapt its learning related parameters such as learning rate
        virtual void PreMinibatchCallback(const Trainer& trainer);

        // TODO: Do we need a separate callback to be called after each minibatch? 

        // Optionally overridable method to get checkpoint state associated with the learner
        virtual Dictionary GetCheckpointState() const;

        // Optionally overridable method to restore the learner's state from a previous checkpoint
        virtual void RestoreFromCheckpoint(const Dictionary& checkpoint);

    public:
        std::vector<Variable> Parameters() const;
    };

    typedef std::shared_ptr<Learner> LearnerPtr;

    // Methods to instantiate CNTK builtin learners
    LearnerPtr SGDLearner(const std::unordered_set<Variable>& parameters, double learningRatePerSample, size_t momentumTimeConstant, bool useNesterovAcceleration = false);
    LearnerPtr AdaGradLearner(const std::unordered_set<Variable>& parameters, size_t momentumTimeConstant, double gaussianNoiseInjectStd);
    LearnerPtr RmsPropLearner(const std::unordered_set<Variable>& parameters, double rmsGamma);
}
