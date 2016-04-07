// A feedforward deep neural network for classification comprised of multiple fully connected layers of hidden representations
// This is representative of the DNNs used until recently for Acoustic modelling in the ASR (automatic speech recognition) pipeline

#include "Trainer.h"
#include <assert.h>
#include <algorithm>
#include "Common.h"

using namespace CNTK;

inline CNTK::FunctionPtr ReLULayer(CNTK::Variable input, size_t outputDim)
{
    assert(input.Shape().size() == 1);
    size_t inputDim = input.Shape()[0];

    auto timesParam = CNTK::Parameter(CNTK::RandomUniform({ outputDim, inputDim }, -0.5, 0.5), L"TimesParam");
    auto timesFunction = CNTK::Times(timesParam, input);

    auto plusParam = CNTK::Parameter(CNTK::Constant({ outputDim }, 0.0), L"BiasParam");
    auto plusFunction = CNTK::Plus(plusParam, timesFunction);

    return CNTK::ReLU(plusFunction);
}

FunctionPtr FullyConnectedFeedForwardClassifierNet(CNTK::Variable input, size_t numOutputClasses, size_t hiddenLayerDim, size_t numHiddenLayers)
{
    assert(numHiddenLayers >= 1);
    FunctionPtr prevReLUFunction = ReLULayer(input, hiddenLayerDim);
    for (size_t i = 1; i < numHiddenLayers; ++i)
        prevReLUFunction = ReLULayer(prevReLUFunction, hiddenLayerDim);

    Variable outputTimesParam = CNTK::Parameter(RandomUniform({ numOutputClasses, hiddenLayerDim }, -0.5, 0.5), L"OutputTimesParam");
    return CNTK::Times(outputTimesParam, prevReLUFunction);
}

std::pair<Value, Value> GetNextMinibatch();

void TrainFeedForwardClassifier()
{
    const size_t inputDim = 937;
    const size_t numOutputClasses = 9404;
    const size_t numHiddenLayers = 6;
    const size_t hiddenLayersDim = 2048;

    Variable inputVar({ inputDim }, L"Features");
    auto classifierOutputFunction = FullyConnectedFeedForwardClassifierNet(inputVar, numOutputClasses, hiddenLayersDim, numHiddenLayers);

    auto labelsVar = Variable({ numOutputClasses }, L"Labels");
    auto trainingLossFunction = CNTK::CrossEntropyWithSoftmax(classifierOutputFunction, labelsVar, L"LossFunction");
    auto predictionFunction = CNTK::PredictionError(classifierOutputFunction, labelsVar, L"PredictionError");


    auto feedForwardClassifier = CNTK::Combined({ trainingLossFunction, predictionFunction }, L"ClassifierModel");

    size_t momentumTimeConstant = 1024;
    double learningRatePerSample = 0.05;
    Trainer feedForwardClassifierTrainer(feedForwardClassifier, trainingLossFunction, { CNTK::SGDLearner(feedForwardClassifier->Parameters(), learningRatePerSample, momentumTimeConstant) }, { trainingLossFunction, predictionFunction });

    Value trainingLossValue, predictionValue;
    
    size_t totalTrainingSampleCount = 100000;
    size_t actualTrainingSampleCount = 0;
    while (actualTrainingSampleCount < totalTrainingSampleCount) 
    {
        auto currentMinibatch = GetNextMinibatch();
        size_t currentMinibatchSize = currentMinibatch.first.Data().Shape().back();
        feedForwardClassifierTrainer.TrainMinibatch({ { inputVar, currentMinibatch.first }, { labelsVar, currentMinibatch.second } });
        actualTrainingSampleCount += currentMinibatchSize;
    }
}
