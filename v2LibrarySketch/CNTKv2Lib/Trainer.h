#pragma once

#include "Function.h"
#include "Learner.h"
#include "Reader.h"
#include "TrainingControl.h"
#include "Distributed.h"

namespace CNTK
{
    class Trainer
    {
    public:
        Trainer(FunctionPtr model, Variable trainingLoss, const std::unordered_set<LearnerPtr>& modelParameterLearners);
        Trainer(FunctionPtr model, Variable trainingLoss, const std::unordered_set<LearnerPtr>& modelParameterLearners, DistributedTrainPtr distributedTrain);

        Trainer(FunctionPtr model, Variable trainingLoss, const std::unordered_set<LearnerPtr>& modelParameterLearners, const std::unordered_set<Variable>& outputs);
        Trainer(FunctionPtr model, Variable trainingLoss, const std::unordered_set<LearnerPtr>& modelParameterLearners, DistributedTrainPtr distributedTrain, const std::unordered_set<Variable>& outputs);

        // Optimizes the model parameters using the specified "arguments" values for model Arguments and returns the computed values for all specified 'outputs' variables 
        // The return value is false if all model parameter learners indicate end of learning (through their Update method's return value)
        bool TrainMinibatch(const std::unordered_map<Variable, const Value>& arguments,
                            const std::unordered_map<Variable, Value>& outputs);

        // Trains the model with data continuously fed by the specified 'reader' and duration of  training determined by the specified TrainingControl object
        // The 'modelArgumentsToReaderStreamMap' argument specifies a 1-1 mapping between the model's argument variables and the reader stream that corresponds to that argument
        void Train(ReaderPtr reader, const std::unordered_map<Variable, StreamDescription>& modelArgumentsToReaderStreamMap, TrainingControlPtr controller);

        size_t NumberOfTrainingSamplesProcessed() const;

        FunctionPtr Model() const;

        Variable TrainingLossVariable() const;
        double LastMinibatchTrainingLoss() const;

        std::unordered_map<Variable, Value> LastMinibatchOutputs() const;

        std::unordered_set<LearnerPtr> Learners() const;

        DistributedTrainPtr DistributedTrain() const;

        void Checkpoint(std::ostream modelStream, std::ostream checkpointStream);
        void RestoreFromCheckpoint(std::istream modelStream, std::istream checkpointStream);
    };
}
