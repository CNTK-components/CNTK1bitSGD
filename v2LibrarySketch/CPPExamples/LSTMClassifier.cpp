// A stacked LSTM network for classification, comprised of multiple stacked LSTM layers with self-stabilization
// This is representative of LSTM models used for Acoustic modelling in the ASR (automatic speech recognition) pipeline

#include "Trainer.h"
#include <assert.h>
#include <algorithm>
#include "Common.h"

using namespace CNTK;

std::pair<FunctionPtr, FunctionPtr> LSTMPCellWithSelfStabilization(Variable input,
                                                                   Variable prevOutput,
                                                                   Variable prevCellState)
{
    assert(input.Shape().size() == 1);
    size_t inputDim = input.Shape()[0];

    size_t outputDim = prevOutput.Shape()[0];
    size_t cellDim = prevCellState.Shape()[0];

    auto Wxo = CNTK::Parameter(CNTK::RandomUniform({ cellDim, inputDim }, -0.5, 0.5), L"WxoParam");
    auto Wxi = CNTK::Parameter(CNTK::RandomUniform({ cellDim, inputDim }, -0.5, 0.5), L"WxiParam");
    auto Wxf = CNTK::Parameter(CNTK::RandomUniform({ cellDim, inputDim }, -0.5, 0.5), L"WxfParam");
    auto Wxc = CNTK::Parameter(CNTK::RandomUniform({ cellDim, inputDim }, -0.5, 0.5), L"WxcParam");

    auto Bo = CNTK::Parameter(CNTK::Constant({ cellDim }, 0.0), L"BoParam");
    auto Bc = CNTK::Parameter(CNTK::Constant({ cellDim }, 0.0), L"BcParam");
    auto Bi = CNTK::Parameter(CNTK::Constant({ cellDim }, 0.0), L"BiParam");
    auto Bf = CNTK::Parameter(CNTK::Constant({ cellDim }, 0.0), L"BfParam");

    auto Whi = CNTK::Parameter(CNTK::RandomUniform({ cellDim, outputDim }, -0.5, 0.5), L"WhiParam");
    auto Wci = CNTK::Parameter(CNTK::RandomUniform({ cellDim }, -0.5, 0.5), L"WciParam");

    auto Whf = CNTK::Parameter(CNTK::RandomUniform({ cellDim, outputDim }, -0.5, 0.5), L"WhfParam");
    auto Wcf = CNTK::Parameter(CNTK::RandomUniform({ cellDim }, -0.5, 0.5), L"WcfParam");

    auto Who = CNTK::Parameter(CNTK::RandomUniform({ cellDim, outputDim }, -0.5, 0.5), L"WhoParam");
    auto Wco = CNTK::Parameter(CNTK::RandomUniform({ cellDim }, -0.5, 0.5), L"WcoParam");

    auto Whc = CNTK::Parameter(CNTK::RandomUniform({ cellDim, outputDim }, -0.5, 0.5), L"WhcParam");

    auto Wmr = CNTK::Parameter(CNTK::RandomUniform({ outputDim, cellDim }, -0.5, 0.5), L"WmrParam");

    // Stabilization by routing input through an extra scalar parameter
    auto sWxo = CNTK::Parameter(0.0, L"sWxoParam");
    auto sWxi = CNTK::Parameter(0.0, L"sWxiParam");
    auto sWxf = CNTK::Parameter(0.0, L"sWxfParam");
    auto sWxc = CNTK::Parameter(0.0, L"sWxcParam");

    auto sWhi = CNTK::Parameter(0.0, L"sWhiParam");
    auto sWci = CNTK::Parameter(0.0, L"sWciParam");

    auto sWhf = CNTK::Parameter(0.0, L"sWhfParam");
    auto sWcf = CNTK::Parameter(0.0, L"sWcfParam");
    auto sWho = CNTK::Parameter(0.0, L"sWhoParam");
    auto sWco = CNTK::Parameter(0.0, L"sWcoParam");
    auto sWhc = CNTK::Parameter(0.0, L"sWhcParam");

    auto sWmr = CNTK::Parameter(0.0, L"sWmrParam");

    auto expsWxo = CNTK::Exp(sWxo);
    auto expsWxi = CNTK::Exp(sWxi);
    auto expsWxf = CNTK::Exp(sWxf);
    auto expsWxc = CNTK::Exp(sWxc);

    auto expsWhi = CNTK::Exp(sWhi);
    auto expsWci = CNTK::Exp(sWci);

    auto expsWhf = CNTK::Exp(sWhf);
    auto expsWcf = CNTK::Exp(sWcf);
    auto expsWho = CNTK::Exp(sWho);
    auto expsWco = CNTK::Exp(sWco);
    auto expsWhc = CNTK::Exp(sWhc);

    auto expsWmr = CNTK::Exp(sWmr);

    auto Wxix = CNTK::Times(Wxi, CNTK::ElementTimes(expsWxi, input));
    auto Whidh = CNTK::Times(Whi, CNTK::ElementTimes(expsWhi, prevOutput));
    auto Wcidc = CNTK::ElementTimes(Wci, CNTK::ElementTimes(expsWci, prevCellState));

    auto it = CNTK::Sigmoid(CNTK::Plus(CNTK::Plus(CNTK::Plus(Wxix, Bi), Whidh), Wcidc));

    auto Wxcx = CNTK::Times(Wxc, CNTK::ElementTimes(expsWxc, input));
    auto Whcdh = CNTK::Times(Whc, CNTK::ElementTimes(expsWhc, prevOutput));
    auto bit = CNTK::ElementTimes(it, CNTK::Tanh(CNTK::Plus(Wxcx, CNTK::Plus(Whcdh, Bc))));

    auto Wxfx = CNTK::Times(Wxf, CNTK::ElementTimes(expsWxf, input));
    auto Whfdh = CNTK::Times(Whf, CNTK::ElementTimes(expsWhf, prevOutput));
    auto Wcfdc = CNTK::ElementTimes(Wcf, CNTK::ElementTimes(expsWcf, prevCellState));

    auto ft = CNTK::Sigmoid(CNTK::Plus(CNTK::Plus(CNTK::Plus(Wxfx, Bf), Whfdh), Wcfdc));

    auto bft = CNTK::ElementTimes(ft, prevCellState);

    auto ct = CNTK::Plus(bft, bit);

    auto Wxox = CNTK::Times(Wxo, CNTK::ElementTimes(expsWxo, input));
    auto Whodh = CNTK::Times(Who, CNTK::ElementTimes(expsWho, prevOutput));
    auto Wcoct = CNTK::ElementTimes(Wco, CNTK::ElementTimes(expsWco, ct));

    auto ot = CNTK::Sigmoid(CNTK::Plus(CNTK::Plus(CNTK::Plus(Wxox, Bo), Whodh), Wcoct));

    auto mt = CNTK::ElementTimes(ot, CNTK::Tanh(ct));

    return { Times(Wmr, ElementTimes(expsWmr, mt)), ct };
}

FunctionPtr LSTMPComponentWithSelfStabilization(Variable input, size_t outputDim, size_t cellDim)
{
    auto outputPlaceholder = Variable({ outputDim }, L"outputPlaceHolder");
    auto dh = PastValue(0.0, outputPlaceholder, L"OutputPastValue");
    auto ctPlaceholder = Variable({ cellDim }, L"ctPlaceHolder");
    auto dc = PastValue(0.0, ctPlaceholder, L"CellPastValue");

    auto LSTMCell = LSTMPCellWithSelfStabilization(input, dh, dc);

    // Form the recurrence loop by connecting the output and cellstate back to the inputs of the respective PastValue nodes
    return Composite(LSTMCell.first, { { outputPlaceholder, LSTMCell.first }, { ctPlaceholder, LSTMCell.second } });
}

FunctionPtr LSTMNet(Variable features, size_t cellDim, size_t hiddenDim, size_t numOutputClasses, size_t numLSTMLayers)
{
    auto nextInput = features;
    for (size_t i = 0; i < numLSTMLayers; ++i) {
        nextInput = LSTMPComponentWithSelfStabilization(nextInput, hiddenDim, cellDim);
    }

    auto W  = CNTK::Parameter(CNTK::RandomUniform({ numOutputClasses, hiddenDim }, -0.5, 0.5), L"OutputWParam");
    auto b = CNTK::Parameter(CNTK::Constant({ numOutputClasses, hiddenDim }, 0.0), L"OutputWParam");

    auto sW = CNTK::Parameter(CNTK::Constant({ 1, 1 }, 0.0), L"sWParam");
    auto expsW = CNTK::Exp(sW);

    return CNTK::Plus(CNTK::Times(W, CNTK::ElementTimes(expsW, nextInput)), b);
}

void TrainLSTMClassifier(MinibatchSourcePtr trainingDataMinibatchSource)
{
    StreamDescription featuresStreamDesc = CNTK::GetStreamDescription(trainingDataMinibatchSource, L"Features");
    const size_t inputDim = featuresStreamDesc.m_sampleLayout[0];

    StreamDescription labelsStreamDesc = CNTK::GetStreamDescription(trainingDataMinibatchSource, L"Labels");
    const size_t numOutputClasses = labelsStreamDesc.m_sampleLayout[0];

    const size_t numLSTMLayers = 3;
    const size_t cellDim = 1024;
    const size_t hiddenDim = 512;

    Variable features({ inputDim }, L"Features");
    auto classifierOutputFunction = LSTMNet(features, cellDim, hiddenDim, numOutputClasses, numLSTMLayers);

    Variable labelsVar = Variable({ numOutputClasses }, L"Labels");
    auto trainingLossFunction = CNTK::CrossEntropyWithSoftmax(classifierOutputFunction, labelsVar, L"lossFunction");
    auto predictionFunction = CNTK::PredictionError(classifierOutputFunction, labelsVar, L"predictionError");

    auto LSTMClassifier = CNTK::Combined({ trainingLossFunction, predictionFunction }, L"LSTMClassifier");

    size_t momentumTimeConstant = 1024;
    double learningRatePerSample = 0.05;

    // Train for 100000 samples; checkpoint every 10000 samples
    TrainingControlPtr driver = CNTK::BasicTrainingControl(100000, 10000, { L"LSTMClassifier.net", L"LSTMClassifier.ckp" });
    Trainer LSTMClassifierTrainer(LSTMClassifier, trainingLossFunction, { CNTK::SGDLearner(LSTMClassifier->Parameters(), learningRatePerSample, momentumTimeConstant) });

    std::unordered_map<Variable, StreamDescription> modelArgumentToMinibatchSourceStreamMap = { { features, featuresStreamDesc }, { labelsVar, labelsStreamDesc } };
    LSTMClassifierTrainer.Train(trainingDataMinibatchSource, modelArgumentToMinibatchSourceStreamMap, driver);
}
