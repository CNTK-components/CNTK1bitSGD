// This example concurrently trains 2 different networks at different update schedules. The first network 
// generates a condition vector for each condition id which is then passed as input to the second AM network which
// uses this condition vector for training the acoustic model. The gradients of the training loss of the AM w.r.t. the
// condition vector input are acculumated over multiple minibatches and when enough samples have been processed for a given
// condition id, the gradients are back propagated to update the learnable parameters of the fist network that computes the
// condition vector for different condition ids based on acoustic data corresponding to the condition id.

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

FunctionPtr AcousticClassiferNet(Variable acousticFeatures, Variable conditionIds, size_t numOutputClasses, size_t hiddenLayerDim, size_t numHiddenLayers, Variable conditionVectors)
{
    auto conditionFeatures = CNTK::Gather(conditionVectors, conditionIds);
    auto augmentedInputFeatures = CNTK::RowStack(acousticFeatures, conditionFeatures);

    assert(numHiddenLayers >= 1);
    FunctionPtr prevReLUFunction = ReLULayer(augmentedInputFeatures, hiddenLayerDim);
    for (size_t i = 1; i < numHiddenLayers; ++i)
        prevReLUFunction = ReLULayer(prevReLUFunction, hiddenLayerDim);

    Variable outputTimesParam = CNTK::Parameter(RandomUniform({ numOutputClasses, hiddenLayerDim }, -0.5, 0.5), L"OutputTimesParam");
    return CNTK::Times(outputTimesParam, prevReLUFunction);
}

FunctionPtr ConditionSummarizationNet(Variable conditionFeatures, size_t conditionVectorDim, size_t hiddenLayerDim, size_t numHiddenLayers)
{
    assert(numHiddenLayers >= 1);
    FunctionPtr prevReLUFunction = ReLULayer(conditionFeatures, hiddenLayerDim);
    for (size_t i = 1; i < numHiddenLayers; ++i)
        prevReLUFunction = ReLULayer(prevReLUFunction, hiddenLayerDim);

    Variable outputTimesParam = CNTK::Parameter(RandomUniform({ conditionVectorDim, hiddenLayerDim }, -0.5, 0.5), L"OutputTimesParam");
    auto conditionOutput = CNTK::Times(outputTimesParam, prevReLUFunction);

    // Now reduce the condition vectors across all samples
    return CNTK::Average(CNTK::Average(conditionOutput, -1 /*reductionAxis*/), BATCH_AXIS /*reductionAxis*/);
}

// Returns a batch of samples of features corresponding to a given condition Id
Value GetConditionFeatures(size_t conditionId);

// Concatenates an array of NDArrayViews to a single NDArrayView by concatenating them along the most significant dimension
NDArrayView ConcatenateArrayViews(const std::vector<NDArrayView>& values);

NDArrayView ComputeConditionVectors(FunctionPtr conditionSummarizationNet, size_t numConditionIds)
{
    std::vector<NDArrayView> conditionVectors(numConditionIds);
    for (size_t i = 0; i < numConditionIds; ++i) {
        conditionSummarizationNet->Forward({ { conditionSummarizationNet->Arguments()[0], GetConditionFeatures(i) } }, { { conditionSummarizationNet, conditionVectors[i] } });
    }

    return ConcatenateArrayViews(conditionVectors);
}

class ConditionVectorsLearner : public Learner
{
public:
    ConditionVectorsLearner(Variable conditionVectorsParam, FunctionPtr conditionSummarizer)
        : Learner({ conditionVectorsParam }),
        m_conditionSummarizerNet(conditionSummarizer),
        m_accumulatedGradients(conditionVectorsParam.Shape(), DataType::Float), m_numSamplesGradientsAccumulatedFor(0)
    {
        m_accumulatedGradients.SetValue<float>(0.0f);

        double learningRatePerSample = 0.05;
        size_t momentumTimeConstant = 1024;
        m_conditionSummarizerParamsLearner = CNTK::SGDLearner(conditionSummarizer->Parameters(), learningRatePerSample, momentumTimeConstant);
    }

    bool Update(const std::unordered_map<Variable, Value>& parameterValues,
                const std::unordered_map<Variable, Value>& gradientValues,
                size_t trainingSampleCount) override
    {
        const size_t updateThreshold = 100000;
        if ((m_numSamplesGradientsAccumulatedFor + trainingSampleCount) > updateThreshold)
        {
            // Backpropagate the accumulated gradients through the conditionSummarizer to update its parameters
            // and generate new conditionVectors to update the conditionVectorsParam
            size_t numConditionIds = m_accumulatedGradients.Shape()[1];
            std::vector<NDArrayView> conditionVectors(numConditionIds);
            bool retVal = false;
            for (size_t i = 0; i < numConditionIds; ++i) {
                BackPropState backpropState = m_conditionSummarizerNet->Forward({ { m_conditionSummarizerNet->Arguments()[0], GetConditionFeatures(i) } }, { { m_conditionSummarizerNet, conditionVectors[i] } }, DeviceDescriptor::DefaultDevice(), true);
                NDArrayView currentConditionIdGradients = gradientValues.begin()->second.Data().Slice(1, i, i + 1);
                std::unordered_map<Variable, Value> paramGradients;
                for (auto param : m_conditionSummarizerNet->Parameters()) {
                    paramGradients[param] = Value();
                }
                m_conditionSummarizerNet->Backward(backpropState, { { m_conditionSummarizerNet, currentConditionIdGradients } }, paramGradients);

                // TODO: Pass the actual number of samples
                retVal |= m_conditionSummarizerParamsLearner->Update(m_conditionSummarizerNet->ParametersValues(), paramGradients, 1);

                // Now update the actual conditionVectorsParam's value
                auto newConditionVectors = ComputeConditionVectors(m_conditionSummarizerNet, numConditionIds);
                parameterValues.begin()->second.Data().CopyFrom(newConditionVectors);
            }

            // Zero accumulated gradients
            m_accumulatedGradients.SetValue<float>(0.0f);
            m_numSamplesGradientsAccumulatedFor = 0;

            return retVal;
        }

        // Accumulate gradients
        AccumulateGradients(gradientValues.begin()->second.Data());
        m_numSamplesGradientsAccumulatedFor += trainingSampleCount;

        return true;
    }

private:
    void AccumulateGradients(NDArrayView gradients);

private:
    FunctionPtr m_conditionSummarizerNet;
    LearnerPtr m_conditionSummarizerParamsLearner;
    NDArrayView m_accumulatedGradients;
    size_t m_numSamplesGradientsAccumulatedFor;
};

std::pair<Value, Value> GetNextMinibatch();

void TrainConditionVectorBasedFeedForwardClassifier()
{
    const size_t inputDim = 937;
    const size_t numOutputClasses = 9404;
    const size_t numHiddenLayers = 6;
    const size_t hiddenLayersDim = 2048;

    size_t conditionSummarizerInputDim = inputDim;
    size_t conditionVectorDim = 300;
    size_t conditionSummarizerHiddenDim = 1024;

    Variable conditionFeatures({ inputDim }, L"ConditionFeatures");
    auto conditionSummarizer = ConditionSummarizationNet(conditionFeatures, conditionVectorDim, conditionSummarizerHiddenDim, 2);

    size_t numConditionIds = 256;
    NDArrayView conditionVectors = ComputeConditionVectors(conditionSummarizer, numConditionIds);
    auto conditionVectorsParam = CNTK::Parameter(conditionVectors, L"ConditionVectors");

    Variable acousticFeatures({ inputDim }, L"AcousticFeatures");
    Variable conditionIds({ 1 }, L"ConditionIds");
    auto classifierOutputFunction = AcousticClassiferNet(acousticFeatures, conditionIds, numOutputClasses, hiddenLayersDim, numHiddenLayers, conditionVectorsParam);

    auto labelsVar = Variable({ numOutputClasses }, L"Labels");
    auto trainingLossFunction = CNTK::CrossEntropyWithSoftmax(classifierOutputFunction, labelsVar, L"LossFunction");
    auto predictionFunction = CNTK::PredictionError(classifierOutputFunction, labelsVar, L"PredictionError");

    auto feedForwardClassifier = CNTK::Combined({ trainingLossFunction, predictionFunction }, L"ClassifierModel");

    size_t momentumTimeConstant = 1024;
    double learningRatePerSample = 0.05;
    std::unordered_set<Variable> nonConditionVectorParams = feedForwardClassifier->Parameters();
    nonConditionVectorParams.erase(conditionVectorsParam);
    auto nonConditionParamsLearner = CNTK::SGDLearner(nonConditionVectorParams, learningRatePerSample, momentumTimeConstant);
    auto conditionVectorsParamsLearner = std::make_shared<ConditionVectorsLearner>(conditionVectorsParam, conditionSummarizer);
    Trainer feedForwardClassifierTrainer(feedForwardClassifier, trainingLossFunction, { nonConditionParamsLearner, conditionVectorsParamsLearner }, { trainingLossFunction, predictionFunction });

    size_t totalTrainingSampleCount = 1000000;
    size_t actualTrainingSampleCount = 0;
    while (actualTrainingSampleCount < totalTrainingSampleCount)
    {
        auto currentMinibatch = GetNextMinibatch();
        size_t currentMinibatchSize = currentMinibatch.first.Data().Shape()[0];
        feedForwardClassifierTrainer.TrainMinibatch({ { acousticFeatures, currentMinibatch.first }, { labelsVar, currentMinibatch.second } });
        actualTrainingSampleCount += currentMinibatchSize;
    }
}
