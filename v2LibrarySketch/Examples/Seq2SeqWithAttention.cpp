// A stacked LSTM network for classification, comprised of multiple stacked LSTM layers with self-stabilization
// This is representative of LSTM models used for Acoustic modelling in the ASR (automatic speech recognition) pipeline

#include "Trainer.h"
#include <assert.h>
#include <algorithm>

using namespace CNTK;

std::pair<FunctionPtr, FunctionPtr> LSTMPCellWithSelfStab(Variable input,
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

    auto sWxo = CNTK::Parameter(CNTK::Constant({ 1, 1 }, 0.0), L"sWxoParam");
    auto sWxi = CNTK::Parameter(CNTK::Constant({ 1, 1 }, 0.0), L"sWxiParam");
    auto sWxf = CNTK::Parameter(CNTK::Constant({ 1, 1 }, 0.0), L"sWxfParam");
    auto sWxc = CNTK::Parameter(CNTK::Constant({ 1, 1 }, 0.0), L"sWxcParam");

    auto sWhi = CNTK::Parameter(CNTK::Constant({ 1, 1 }, 0.0), L"sWhiParam");
    auto sWci = CNTK::Parameter(CNTK::Constant({ 1, 1 }, 0.0), L"sWciParam");

    auto sWhf = CNTK::Parameter(CNTK::Constant({ 1, 1 }, 0.0), L"sWhfParam");
    auto sWcf = CNTK::Parameter(CNTK::Constant({ 1, 1 }, 0.0), L"sWcfParam");
    auto sWho = CNTK::Parameter(CNTK::Constant({ 1, 1 }, 0.0), L"sWhoParam");
    auto sWco = CNTK::Parameter(CNTK::Constant({ 1, 1 }, 0.0), L"sWcoParam");
    auto sWhc = CNTK::Parameter(CNTK::Constant({ 1, 1 }, 0.0), L"sWhcParam");

    auto sWmr = CNTK::Parameter(CNTK::Constant({ 1, 1 }, 0.0), L"sWmrParam");

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

    auto Wxix = CNTK::Times(Wxi, CNTK::Scale(expsWxi, input));
    auto Whidh = CNTK::Times(Whi, CNTK::Scale(expsWhi, prevOutput));
    auto Wcidc = CNTK::DiagTimes(Wci, CNTK::Scale(expsWci, prevCellState));

    auto it = CNTK::Sigmoid(CNTK::Plus(CNTK::Plus(CNTK::Plus(Wxix, Bi), Whidh), Wcidc));

    auto Wxcx = CNTK::Times(Wxc, CNTK::Scale(expsWxc, input));
    auto Whcdh = CNTK::Times(Whc, CNTK::Scale(expsWhc, prevOutput));
    auto bit = CNTK::ElementTimes(it, CNTK::Tanh(CNTK::Plus(Wxcx, CNTK::Plus(Whcdh, Bc))));

    auto Wxfx = CNTK::Times(Wxf, CNTK::Scale(expsWxf, input));
    auto Whfdh = CNTK::Times(Whf, CNTK::Scale(expsWhf, prevOutput));
    auto Wcfdc = CNTK::DiagTimes(Wcf, CNTK::Scale(expsWcf, prevCellState));

    auto ft = CNTK::Sigmoid(CNTK::Plus(CNTK::Plus(CNTK::Plus(Wxfx, Bf), Whfdh), Wcfdc));

    auto bft = CNTK::ElementTimes(ft, prevCellState);

    auto ct = CNTK::Plus(bft, bit);

    auto Wxox = CNTK::Times(Wxo, CNTK::Scale(expsWxo, input));
    auto Whodh = CNTK::Times(Who, CNTK::Scale(expsWho, prevOutput));
    auto Wcoct = CNTK::DiagTimes(Wco, CNTK::Scale(expsWco, ct));

    auto ot = CNTK::Sigmoid(CNTK::Plus(CNTK::Plus(CNTK::Plus(Wxox, Bo), Whodh), Wcoct));

    auto mt = CNTK::ElementTimes(ot, CNTK::Tanh(ct));

    return { CNTK::Times(Wmr, CNTK::Scale(expsWmr, mt)), ct };
}

FunctionPtr EncoderSubNet(size_t inputDim, size_t cellDim, size_t hiddenDim)
{
    Variable sourceInput({ inputDim }, L"Source");

    auto outputPlaceholder = Variable({ hiddenDim }, L"outputPlaceHolder");
    auto dh = CNTK::PastValue(CNTK::Constant(outputPlaceholder.Shape(), 0.0), outputPlaceholder, L"OutputPastValue");
    auto ctPlaceholder = Variable({ cellDim }, L"ctPlaceHolder");
    auto dc = CNTK::PastValue(CNTK::Constant(ctPlaceholder.Shape(), 0.0), ctPlaceholder, L"CellPastValue");

    auto LSTMCell = LSTMPCellWithSelfStab(sourceInput, dh, dc);

    // Form the recurrence loop by connecting the output and cellstate back to the inputs of the respective PastValue nodes
    return CNTK::Composite(LSTMCell.first, { { outputPlaceholder, LSTMCell.first }, { ctPlaceholder, LSTMCell.second } });
}

// Attention as described in http://arxiv.org/pdf/1412.7449v3.pdf
FunctionPtr Attention(Variable encoderStates, Variable decoderState)
{
    size_t encoderStateDim = encoderStates.Shape()[0];
    size_t decoderStateDim = decoderState.Shape()[0];
    size_t attentionDim = decoderStateDim;
    auto encoderStateProjParams = CNTK::Parameter(CNTK::RandomUniform({ attentionDim, encoderStateDim }, -0.5, 0.5), L"AttentionEncoderParams");
    auto encoderProj = CNTK::Times(encoderStateProjParams, encoderStates);
    auto decoderStateProjParams = CNTK::Parameter(CNTK::RandomUniform({ attentionDim, decoderStateDim }, -0.5, 0.5), L"AttentionDecoderParams");
    auto decoderProj = CNTK::Times(decoderStateProjParams, decoderState);

    auto reductionParams = CNTK::Parameter(CNTK::RandomUniform({ 1, attentionDim }, -0.5, 0.5), L"AttentionReductionParams");
    // The Plus operation below broadcasts along the column dimension of the projected encoder state
    auto u = CNTK::Times(reductionParams, CNTK::Tanh(CNTK::Plus(decoderProj, encoderProj)));

    // Perform a Softmax along the column dimension of 'u' to obtain the weight vector for the attention
    auto attentionWeights = CNTK::Softmax(u, 1 /*axis*/);

    return CNTK::Times(encoderStates, attentionWeights);
}

// Note that this is just for training where we use an input target sentence to drive the decoder instead of its own output from previous step
FunctionPtr DecoderWithAttention(Variable encoderStates, size_t outputDim, size_t cellDim)
{
    // The decoder is a recurrent network that uses attention over the hidden states of the encoder emitted for each step of the source sequence

    Variable targetInput({ outputDim }, L"Target");

    // Compute the context vector using attention over the encoderStates
    auto ctPlaceholder = Variable({ cellDim }, L"ctPlaceHolder");
    auto dc = CNTK::PastValue(CNTK::Constant(ctPlaceholder.Shape(), 0.0), ctPlaceholder, L"CellPastValue");

    auto context = Attention(encoderStates, dc);
    auto LSTMCell = LSTMPCellWithSelfStab(context, targetInput, dc);

    // Form the recurrence loop by connecting the cellstate back to the input of the PastValue node
    return CNTK::Composite(LSTMCell.first, { { ctPlaceholder, LSTMCell.second } });
}

FunctionPtr EncoderDecoderWithAttention(size_t inputDim, size_t cellDim, size_t hiddenDim, size_t outputDim)
{
    auto encoderFunction = EncoderSubNet(inputDim, cellDim, hiddenDim);

    // Each Variable has a Shape (denoting the shape of a sample) and one implicit sequence dimension(s) denoting the fixed or variable lengths of
    // the sequence that the Variable denotes.
    // We reshape the encoder state sequence to fold the sequence axis into the sample shape so that it turns from being a sequence of states to
    // a single sample consisting of variable number of columns corresponding to the length of the source sequence. We then perform batch algebraic and reduction
    // operations over the encoder state inside the decoder recurrence loop.
    NDShape newShape = { 1, INFERRED_DIMENSION };
    newShape.insert(newShape.end(), encoderFunction->Output().Shape().begin(), encoderFunction->Output().Shape().end());
    auto encoderStates = CNTK::Reshape(encoderFunction, -1, (int)encoderFunction->Output().Shape().size(), newShape);

    return DecoderWithAttention(encoderStates, outputDim, cellDim);
}

void TrainEncoderDecoder(ReaderPtr trainingDataReader)
{
    StreamDescription sourceSequenceStreamDesc = CNTK::GetStreamDescription(trainingDataReader, L"Source");
    const size_t inputDim = sourceSequenceStreamDesc.m_sampleLayout[0];

    StreamDescription targetSequenceStreamDesc = CNTK::GetStreamDescription(trainingDataReader, L"Target");
    const size_t outputDim = targetSequenceStreamDesc.m_sampleLayout[0];

    const size_t cellDim = 1024;
    const size_t hiddenDim = 512;
    auto encoderDecoderNetOutputFunction = EncoderDecoderWithAttention(inputDim, cellDim, hiddenDim, outputDim);

    Variable labelsVar = Variable({ outputDim }, L"Target");
    auto trainingLossFunction = CNTK::CrossEntropyWithSoftmax(encoderDecoderNetOutputFunction, labelsVar, L"lossFunction");
    auto trainingNet = trainingLossFunction;

    size_t momentumTimeConstant = 1024;
    double learningRatePerSample = 0.05;

    // Train for 100000 samples; checkpoint every 10000 samples
    TrainingControlPtr driver = CNTK::BasicTrainingControl(100000, 10000, { L"EncoderDecoderWithAttn.net", L"EncoderDecoderWithAttn.ckp" });
    Trainer encoderDecoderTrainer(trainingNet, trainingLossFunction, { CNTK::SGDLearner(trainingNet->Parameters(), learningRatePerSample, momentumTimeConstant) });

    std::unordered_map<Variable, StreamDescription> modelArgumentToReaderStreamMap = { { encoderDecoderNetOutputFunction->Argument(), sourceSequenceStreamDesc }, { labelsVar, targetSequenceStreamDesc } };
    encoderDecoderTrainer.Train(trainingDataReader, modelArgumentToReaderStreamMap, driver);
}
