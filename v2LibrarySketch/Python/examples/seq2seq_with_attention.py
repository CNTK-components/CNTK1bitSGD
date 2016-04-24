#
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
#

# A stacked LSTM network for classification, comprised of multiple stacked LSTM layers with self-stabilization
# This is representative of LSTM models used for Acoustic modelling in the ASR (automatic speech recognition) pipeline

from ..cntk import *

def LSTMP_cell_with_self_stabilization(input: Variable, prevOutput: Variable, prevCellState: Variable) -> Tuple[Function, Function]:
    assert(len(input.shape()) == 1)
    inputDim = input.shape()[0]

    outputDim = prevOutput.shape()[0]
    cellDim = prevCellState.shape()[0]

    Wxo = parameter(random_uniform([ cellDim, inputDim ], -0.5, 0.5), "WxoParam")
    Wxi = parameter(random_uniform([ cellDim, inputDim ], -0.5, 0.5), "WxiParam")
    Wxf = parameter(random_uniform([ cellDim, inputDim ], -0.5, 0.5), "WxfParam")
    Wxc = parameter(random_uniform([ cellDim, inputDim ], -0.5, 0.5), "WxcParam")

    Bo = parameter(constant([ cellDim ], 0.0), "BoParam")
    Bc = parameter(constant([ cellDim ], 0.0), "BcParam")
    Bi = parameter(constant([ cellDim ], 0.0), "BiParam")
    Bf = parameter(constant([ cellDim ], 0.0), "BfParam")

    Whi = parameter(random_uniform([ cellDim, outputDim ], -0.5, 0.5), "WhiParam")
    Wci = parameter(random_uniform([ cellDim ], -0.5, 0.5), "WciParam")

    Whf = parameter(random_uniform([ cellDim, outputDim ], -0.5, 0.5), "WhfParam")
    Wcf = parameter(random_uniform([ cellDim ], -0.5, 0.5), "WcfParam")

    Who = parameter(random_uniform([ cellDim, outputDim ], -0.5, 0.5), "WhoParam")
    Wco = parameter(random_uniform([ cellDim ], -0.5, 0.5), "WcoParam")

    Whc = parameter(random_uniform([ cellDim, outputDim ], -0.5, 0.5), "WhcParam")

    Wmr = parameter(random_uniform([ outputDim, cellDim ], -0.5, 0.5), "WmrParam")

    sWxo = parameter(0.0, "sWxoParam")
    sWxi = parameter(0.0, "sWxiParam")
    sWxf = parameter(0.0, "sWxfParam")
    sWxc = parameter(0.0, "sWxcParam")

    sWhi = parameter(0.0, "sWhiParam")
    sWci = parameter(0.0, "sWciParam")

    sWhf = parameter(0.0, "sWhfParam")
    sWcf = parameter(0.0, "sWcfParam")
    sWho = parameter(0.0, "sWhoParam")
    sWco = parameter(0.0, "sWcoParam")
    sWhc = parameter(0.0, "sWhcParam")

    sWmr = parameter(0.0, "sWmrParam")

    expsWxo = exp(sWxo)
    expsWxi = exp(sWxi)
    expsWxf = exp(sWxf)
    expsWxc = exp(sWxc)

    expsWhi = exp(sWhi)
    expsWci = exp(sWci)

    expsWhf = exp(sWhf)
    expsWcf = exp(sWcf)
    expsWho = exp(sWho)
    expsWco = exp(sWco)
    expsWhc = exp(sWhc)

    expsWmr = exp(sWmr)

    Wxix = times(Wxi, expsWxi * input)
    Whidh = times(Whi, expsWhi * prevOutput)
    Wcidc = Wci * (expsWci * prevCellState)

    it = sigmoid((((Wxix + Bi) + Whidh) + Wcidc))

    Wxcx = times(Wxc, expsWxc * input)
    Whcdh = times(Whc, expsWhc * prevOutput)
    bit = it * tanh(Wxcx + (Whcdh + Bc))

    Wxfx = times(Wxf, expsWxf * input)
    Whfdh = times(Whf, expsWhf * prevOutput)
    Wcfdc = Wcf * (expsWcf * prevCellState)

    ft = sigmoid((((Wxfx + Bf) + Whfdh) + Wcfdc))

    bft = ft * prevCellState

    ct = bft + bit

    Wxox = times(Wxo, expsWxo * input)
    Whodh = times(Who, expsWho * prevOutput)
    Wcoct = Wco * (expsWco * ct)

    ot = sigmoid((((Wxox + Bo) + Whodh) + Wcoct))

    mt = ot * tanh(ct)

    return ( times(Wmr, (expsWmr * mt)), ct )

def encoder_subnet(sourceInput: Variable, cellDim: int, hiddenDim: int) -> Function:
    outputPlaceholder = Variable([ hiddenDim ], "outputPlaceHolder")
    dh = past_value(0.0, outputPlaceholder, "OutputPastValue")
    ctPlaceholder = Variable([ cellDim ], "ctPlaceHolder")
    dc = past_value(0.0, ctPlaceholder, "CellPastValue")

    LSTMCell = LSTMP_cell_with_self_stabilization(sourceInput, dh, dc)

    # Form the recurrence loop by connecting the output and cellstate back to the inputs of the respective past_value nodes
    return composite(LSTMCell[0], {outputPlaceholder : LSTMCell[0], ctPlaceholder : LSTMCell[1]})

# Attention as described in http://arxiv.org/pdf/1412.7449v3.pdf
def attention(encoderState: Variable, decoderState: Variable) -> Function:
    attentionDim = decoderState.shape()
    encoderStateProjParams = parameter(random_uniform(attentionDim.append_shape(encoderState.shape()), -0.5, 0.5), "AttentionEncoderParams")
    encoderProj = times(encoderStateProjParams, encoderState)
    decoderStateProjParams = parameter(random_uniform(attentionDim.append_shape(decoderState.shape()), -0.5, 0.5), "AttentionDecoderParams")
    decoderProj = times(decoderStateProjParams, decoderState)

    reductionParams = parameter(random_uniform(NDShape([ 1 ]).append_shape(attentionDim), -0.5, 0.5), "AttentionReductionParams")

    # The Plus operation below broadcasts along the column dimension of the projected encoder state
    u = times(reductionParams, tanh(decoderProj + encoderProj))

    # Perform a Softmax along the sequence axis of encoderState to obtain the weight vector for the attention
    encoderStateDynamicAxis = encoderState.dynamic_axes()[0]
    attentionWeights = softmax(u, encoderStateDynamicAxis)

    # Now we multiply the encoderState with the attention state and then sum along the sequence axis of encoderState
    return sum(attentionWeights * encoderState, encoderStateDynamicAxis)

# Note that this is just for training where we use an input target sentence to drive the decoder instead of its own output from previous step
def decoder_with_attention(targetInput: Variable, encoderStates: Variable, cellDim: int) -> Function:
    # The decoder is a recurrent network that uses attention over the hidden states of the encoder for each step of the source sequence

    # Compute the context vector using attention over the encoderStates
    outputPlaceholder = Variable([ INFERRED_DIMENSION ], "outputPlaceholder")
    dh = past_value(0.0, outputPlaceholder, "OutputPastValue")

    context = attention(encoderStates, dh)
    LSTMCell = LSTMP_cell_with_self_stabilization(context, targetInput, dh)

    # Form the recurrence loop by connecting the cellstate back to the input of the past_value node
    return composite(LSTMCell[0], {outputPlaceholder : LSTMCell[0]})

def encoder_decoder_with_attention(sourceInput: Variable, targetInput: Variable, cellDim: int, hiddenDim: int) -> Function:
    encoderFunction = encoder_subnet(sourceInput, cellDim, hiddenDim)
    return decoder_with_attention(targetInput, encoderFunction, cellDim)

def train_encoder_decoder(trainingDataMinibatchSource: MinibatchSource):
    sourceSequenceStreamDesc = get_stream_description(trainingDataMinibatchSource, "Source")
    inputDim = sourceSequenceStreamDesc.m_sampleLayout[0]
    sourceInput = Variable([ inputDim ], AxisId.new_dynamic_axis("Source"), "Source")

    targetSequenceStreamDesc = get_stream_description(trainingDataMinibatchSource, "Target")
    outputDim = targetSequenceStreamDesc.m_sampleLayout[0]
    targetInput = Variable([ outputDim ], AxisId.new_dynamic_axis("Target"), "Target")

    cellDim = 1024
    hiddenDim = 512
    encoderDecoderNetOutputFunction = encoder_decoder_with_attention(sourceInput, targetInput, cellDim, hiddenDim)

    trainingLossFunction = cross_entropy_with_softmax(encoderDecoderNetOutputFunction, targetInput, "lossFunction")
    trainingNet = trainingLossFunction

    momentumTimeConstant = 1024
    learningRatePerSample = 0.05

    # Train for 100000 samples checkpoint every 10000 samples
    driver = basic_training_control(100000, 10000, [ "EncoderDecoderWithAttn.net", "EncoderDecoderWithAttn.ckp" ])
    encoderDecoderTrainer = Trainer(trainingNet, trainingLossFunction, {sgd_learner(trainingNet.parameters(), learningRatePerSample, momentumTimeConstant)})

    modelArgumentToMinibatchSourceStreamMap = {sourceInput : sourceSequenceStreamDesc, targetInput : targetSequenceStreamDesc}
    encoderDecoderTrainer.train(trainingDataMinibatchSource, modelArgumentToMinibatchSourceStreamMap, driver)
