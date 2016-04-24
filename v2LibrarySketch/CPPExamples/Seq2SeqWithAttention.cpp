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

    auto Wxo = Parameter(RandomUniform({ cellDim, inputDim }, -0.5, 0.5), L"WxoParam");
    auto Wxi = Parameter(RandomUniform({ cellDim, inputDim }, -0.5, 0.5), L"WxiParam");
    auto Wxf = Parameter(RandomUniform({ cellDim, inputDim }, -0.5, 0.5), L"WxfParam");
    auto Wxc = Parameter(RandomUniform({ cellDim, inputDim }, -0.5, 0.5), L"WxcParam");

    auto Bo = Parameter(Constant({ cellDim }, 0.0), L"BoParam");
    auto Bc = Parameter(Constant({ cellDim }, 0.0), L"BcParam");
    auto Bi = Parameter(Constant({ cellDim }, 0.0), L"BiParam");
    auto Bf = Parameter(Constant({ cellDim }, 0.0), L"BfParam");

    auto Whi = Parameter(RandomUniform({ cellDim, outputDim }, -0.5, 0.5), L"WhiParam");
    auto Wci = Parameter(RandomUniform({ cellDim }, -0.5, 0.5), L"WciParam");

    auto Whf = Parameter(RandomUniform({ cellDim, outputDim }, -0.5, 0.5), L"WhfParam");
    auto Wcf = Parameter(RandomUniform({ cellDim }, -0.5, 0.5), L"WcfParam");

    auto Who = Parameter(RandomUniform({ cellDim, outputDim }, -0.5, 0.5), L"WhoParam");
    auto Wco = Parameter(RandomUniform({ cellDim }, -0.5, 0.5), L"WcoParam");

    auto Whc = Parameter(RandomUniform({ cellDim, outputDim }, -0.5, 0.5), L"WhcParam");

    auto Wmr = Parameter(RandomUniform({ outputDim, cellDim }, -0.5, 0.5), L"WmrParam");

    auto sWxo = Parameter(0.0, L"sWxoParam");
    auto sWxi = Parameter(0.0, L"sWxiParam");
    auto sWxf = Parameter(0.0, L"sWxfParam");
    auto sWxc = Parameter(0.0, L"sWxcParam");

    auto sWhi = Parameter(0.0, L"sWhiParam");
    auto sWci = Parameter(0.0, L"sWciParam");

    auto sWhf = Parameter(0.0, L"sWhfParam");
    auto sWcf = Parameter(0.0, L"sWcfParam");
    auto sWho = Parameter(0.0, L"sWhoParam");
    auto sWco = Parameter(0.0, L"sWcoParam");
    auto sWhc = Parameter(0.0, L"sWhcParam");

    auto sWmr = Parameter(0.0, L"sWmrParam");

    auto expsWxo = Exp(sWxo);
    auto expsWxi = Exp(sWxi);
    auto expsWxf = Exp(sWxf);
    auto expsWxc = Exp(sWxc);

    auto expsWhi = Exp(sWhi);
    auto expsWci = Exp(sWci);

    auto expsWhf = Exp(sWhf);
    auto expsWcf = Exp(sWcf);
    auto expsWho = Exp(sWho);
    auto expsWco = Exp(sWco);
    auto expsWhc = Exp(sWhc);

    auto expsWmr = Exp(sWmr);

    auto Wxix = Times(Wxi, expsWxi * input);
    auto Whidh = Times(Whi, expsWhi * prevOutput);
    auto Wcidc = Wci * (expsWci * prevCellState);

    auto it = Sigmoid((((Wxix + Bi) + Whidh) + Wcidc));

    auto Wxcx = Times(Wxc, expsWxc * input);
    auto Whcdh = Times(Whc, expsWhc * prevOutput);
    auto bit = it * Tanh(Wxcx + (Whcdh + Bc));

    auto Wxfx = Times(Wxf, expsWxf * input);
    auto Whfdh = Times(Whf, expsWhf * prevOutput);
    auto Wcfdc = Wcf * (expsWcf * prevCellState);

    auto ft = Sigmoid((((Wxfx + Bf) + Whfdh) + Wcfdc));

    auto bft = ft * prevCellState;

    auto ct = bft + bit;

    auto Wxox = Times(Wxo, expsWxo * input);
    auto Whodh = Times(Who, expsWho * prevOutput);
    auto Wcoct = Wco * (expsWco * ct);

    auto ot = Sigmoid((((Wxox + Bo) + Whodh) + Wcoct));

    auto mt = ot * Tanh(ct);

    return { Times(Wmr, (expsWmr * mt)), ct };
}

FunctionPtr EncoderSubNet(Variable sourceInput, size_t cellDim, size_t hiddenDim)
{
    auto outputPlaceholder = Variable({ hiddenDim }, L"outputPlaceHolder");
    auto dh = PastValue(0.0, outputPlaceholder, L"OutputPastValue");
    auto ctPlaceholder = Variable({ cellDim }, L"ctPlaceHolder");
    auto dc = PastValue(0.0, ctPlaceholder, L"CellPastValue");

    auto LSTMCell = LSTMPCellWithSelfStabilization(sourceInput, dh, dc);

    // Form the recurrence loop by connecting the output and cellstate back to the inputs of the respective PastValue nodes
    return Composite(LSTMCell.first, { { outputPlaceholder, LSTMCell.first }, { ctPlaceholder, LSTMCell.second } });
}

// Attention as described in http://arxiv.org/pdf/1412.7449v3.pdf
FunctionPtr Attention(Variable encoderState, Variable decoderState)
{
    NDShape attentionDim = decoderState.Shape();
    auto encoderStateProjParams = Parameter(RandomUniform(attentionDim.AppendShape(encoderState.Shape()), -0.5, 0.5), L"AttentionEncoderParams");
    auto encoderProj = Times(encoderStateProjParams, encoderState);
    auto decoderStateProjParams = Parameter(RandomUniform(attentionDim.AppendShape(decoderState.Shape()), -0.5, 0.5), L"AttentionDecoderParams");
    auto decoderProj = Times(decoderStateProjParams, decoderState);

    auto reductionParams = Parameter(RandomUniform(NDShape({ 1 }).AppendShape(attentionDim), -0.5, 0.5), L"AttentionReductionParams");

    // The + operation below broadcasts along the column dimension of the projected encoder state
    auto u = Times(reductionParams, Tanh(decoderProj + encoderProj));

    // Perform a Softmax along the sequence axis of encoderState to obtain the weight vector for the attention
    AxisId encoderStateDynamicAxis = *encoderState.DynamicAxes().begin();
    auto attentionWeights = Softmax(u, encoderStateDynamicAxis);

    // Now we multiply the encoderState with the attention state and then sum along the sequence axis of encoderState
    return Sum(attentionWeights * encoderState, encoderStateDynamicAxis);
}

// Note that this is just for training where we use an input target sentence to drive the decoder instead of its own output from previous step
FunctionPtr DecoderWithAttention(Variable targetInput, Variable encoderStates, size_t cellDim)
{
    // The decoder is a recurrent network that uses attention over the hidden states of the encoder for each step of the source sequence

    // Compute the context vector using attention over the encoderStates
    auto outputPlaceholder = Variable({ INFERRED_DIMENSION }, L"outputPlaceholder");
    auto dh = PastValue(0.0, outputPlaceholder, L"OutputPastValue");

    auto context = Attention(encoderStates, dh);
    auto LSTMCell = LSTMPCellWithSelfStabilization(context, targetInput, dh);

    // Form the recurrence loop by connecting the cellstate back to the input of the PastValue node
    return Composite(LSTMCell.first, { { outputPlaceholder, LSTMCell.first } });
}

FunctionPtr EncoderDecoderWithAttention(Variable sourceInput, Variable targetInput, size_t cellDim, size_t hiddenDim)
{
    auto encoderFunction = EncoderSubNet(sourceInput, cellDim, hiddenDim);
    return DecoderWithAttention(targetInput, encoderFunction, cellDim);
}

void TrainEncoderDecoder(MinibatchSourcePtr trainingDataMinibatchSource)
{
    StreamDescription sourceSequenceStreamDesc = GetStreamDescription(trainingDataMinibatchSource, L"Source");
    const size_t inputDim = sourceSequenceStreamDesc.m_sampleLayout[0];
    Variable sourceInput({ inputDim }, AxisId::NewDynamicAxis(L"Source"), L"Source");

    StreamDescription targetSequenceStreamDesc = GetStreamDescription(trainingDataMinibatchSource, L"Target");
    const size_t outputDim = targetSequenceStreamDesc.m_sampleLayout[0];
    Variable targetInput({ outputDim }, AxisId::NewDynamicAxis(L"Target"), L"Target");

    const size_t cellDim = 1024;
    const size_t hiddenDim = 512;
    auto encoderDecoderNetOutputFunction = EncoderDecoderWithAttention(sourceInput, targetInput, cellDim, hiddenDim);

    auto trainingLossFunction = CrossEntropyWithSoftmax(encoderDecoderNetOutputFunction, targetInput, L"lossFunction");
    auto trainingNet = trainingLossFunction;

    size_t momentumTimeConstant = 1024;
    double learningRatePerSample = 0.05;

    // Train for 100000 samples; checkpoint every 10000 samples
    TrainingControlPtr driver = BasicTrainingControl(100000, 10000, { L"EncoderDecoderWithAttn.net", L"EncoderDecoderWithAttn.ckp" });
    Trainer encoderDecoderTrainer(trainingNet, trainingLossFunction, { SGDLearner(trainingNet->Parameters(), learningRatePerSample, momentumTimeConstant) });

    std::unordered_map<Variable, StreamDescription> modelArgumentToMinibatchSourceStreamMap = { { sourceInput, sourceSequenceStreamDesc }, { targetInput, targetSequenceStreamDesc } };
    encoderDecoderTrainer.Train(trainingDataMinibatchSource, modelArgumentToMinibatchSourceStreamMap, driver);
}
