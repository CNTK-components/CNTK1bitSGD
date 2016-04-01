// A 34 layer deep residual convolutional network for image classification as described in http://arxiv.org/pdf/1512.03385v1.pdf

#include "Trainer.h"
#include <assert.h>
#include <algorithm>

using namespace CNTK;

FunctionPtr ConvBNLayer(Variable input, size_t featureMapCount, size_t kernelWidth, size_t kernelHeight, size_t hStride, size_t vStride, double wScale, double bValue, double scValue, size_t bnTimeConst)
{
    size_t numInputChannels = input.Shape().back();
    auto convParams = CNTK::Parameter(CNTK::RandomNormal({ featureMapCount, kernelWidth, kernelHeight, numInputChannels }, 0.0, wScale));
    auto convFunction = CNTK::Convolution(convParams, input, { hStride, vStride }, true /*zeroPadding*/);

    auto biasParams = CNTK::Parameter(CNTK::Constant({ featureMapCount }, bValue));
    auto scaleParams = CNTK::Parameter(CNTK::Constant({ featureMapCount }, scValue));
    auto runningMean = CNTK::Parameter(CNTK::Constant({ featureMapCount }, 0.0));
    auto runningInvStd = CNTK::Parameter(CNTK::Constant({ featureMapCount }, 0.0));
    return CNTK::BatchNormalization(convFunction, scaleParams, biasParams, runningMean, runningInvStd, true /*spatial*/, bnTimeConst, 0.000000001 /* epsilon */);
}

FunctionPtr ConvBNReLULayer(Variable input, size_t featureMapCount, size_t kernelWidth, size_t kernelHeight, size_t hStride, size_t vStride, double wScale, double bValue, double scValue, size_t bnTimeConst)
{
    auto convBNFunction = ConvBNLayer(input, featureMapCount, kernelWidth, kernelHeight, hStride, vStride, wScale, bValue, scValue, bnTimeConst);
    return CNTK::ReLU(convBNFunction);
}

// Standard building block for ResNet with identity shortcut(option A).
FunctionPtr ResNetNode2A(Variable input, size_t featureMapCount, size_t kernelWidth, size_t kernelHeight, double wScale, double bValue, double scValue, size_t bnTimeConst)
{
    auto conv1 = ConvBNReLULayer(input, featureMapCount, kernelWidth, kernelHeight, 1, 1, wScale, bValue, scValue, bnTimeConst);
    auto conv2 = ConvBNLayer(conv1, featureMapCount, kernelWidth, kernelHeight, 1, 1, wScale, bValue, scValue, bnTimeConst);

    // Identity shortcut followed by ReLU.
    return CNTK::ReLU(CNTK::Plus(conv2, input));
}

// Standard building block for ResNet with padding(option B).
FunctionPtr ResNetNode2BInc(Variable input, size_t outFeatureMapCount, size_t kernelWidth, size_t kernelHeight, double wScale, double bValue, double scValue, size_t bnTimeConst)
{
    size_t numInputChannels = input.Shape().back();
    auto conv1 = ConvBNReLULayer(input, outFeatureMapCount, kernelWidth, kernelHeight, 2, 2, wScale, bValue, scValue, bnTimeConst);
    auto conv2 = ConvBNLayer(conv1, outFeatureMapCount, kernelWidth, kernelHeight, 1, 1, wScale, bValue, scValue, bnTimeConst);

    // Projection convolution layer.
    auto cProj = ConvBNLayer(input, outFeatureMapCount, 1, 1, 2, 2, wScale, bValue, scValue, bnTimeConst);
    return CNTK::ReLU(CNTK::Plus(conv2, cProj));
}

FunctionPtr ResNet34ClassifierNet(const NDShape& inputImageShape, size_t numOutputClasses)
{
    double conv1WScale = 0.6;
    double convBValue = 0;
    double scValue = 1;
    size_t cMap1 = 64;
    size_t bnTimeConst = 4096;
    Variable imageInput(inputImageShape, L"Images");
    auto conv1 = ConvBNReLULayer(imageInput, cMap1, 7, 7, 2, 2, conv1WScale, convBValue, scValue, bnTimeConst);

    // Max pooling
    size_t pool1W = 3;
    size_t pool1H = 3;
    size_t pool1hs = 2;
    size_t pool1vs = 2;
    auto rootFunction = CNTK::Pooling(conv1, PoolingType::Max, { pool1W, pool1H }, { pool1hs, pool1vs }, { true, true, false } /* autoPaddding */);

    // Initial parameter values.
    double convWScale = 7.07;
    size_t kW = 3;
    size_t kH = 3;
    size_t hs = 1;
    size_t vs = 1;

    for (int i = 0; i < 3; ++i)
        rootFunction = ResNetNode2A(rootFunction, cMap1, kW, kH, convWScale, convBValue, scValue, bnTimeConst);

    size_t cMap2 = 128;
    rootFunction = ResNetNode2BInc(rootFunction, cMap2, kW, kH, convWScale, convBValue, scValue, bnTimeConst);
    for (int i = 0; i < 3; ++i)
        rootFunction = ResNetNode2A(rootFunction, cMap2, kW, kH, convWScale, convBValue, scValue, bnTimeConst);

    size_t cMap3 = 256;
    rootFunction = ResNetNode2BInc(rootFunction, cMap3, kW, kH, convWScale, convBValue, scValue, bnTimeConst);
    for (int i = 0; i < 5; ++i)
        rootFunction = ResNetNode2A(rootFunction, cMap3, kW, kH, convWScale, convBValue, scValue, bnTimeConst);

    size_t cMap4 = 512;
    rootFunction = ResNetNode2BInc(rootFunction, cMap4, kW, kH, convWScale, convBValue, scValue, bnTimeConst);
    for (int i = 0; i < 2; ++i)
        rootFunction = ResNetNode2A(rootFunction, cMap4, kW, kH, convWScale, convBValue, scValue, bnTimeConst);

    // Global average pooling
    size_t pool2W = 7;
    size_t pool2H = 7;
    size_t pool2hs = 1;
    size_t pool2vs = 1;
    rootFunction = CNTK::Pooling(rootFunction, PoolingType::Average, { pool2W, pool2H }, { pool2hs, pool2vs });

    // Output DNN layer
    double fcWScale = 1.13;
    auto outTimesParams = CNTK::Parameter(CNTK::RandomNormal({ cMap4, numOutputClasses }, 0, fcWScale));
    auto outBiasParams = CNTK::Parameter(CNTK::Constant({ numOutputClasses }, 0.0));
    
    return CNTK::Plus(CNTK::Times(outTimesParams, rootFunction), outBiasParams);
}

void TrainImageClassifier(ReaderPtr imageReader)
{
    StreamDescription imagesStreamDesc = CNTK::GetStreamDescription(imageReader, L"Images");
    auto inputImageShape = imagesStreamDesc.m_sampleLayout;

    StreamDescription labelsStreamDesc = CNTK::GetStreamDescription(imageReader, L"Labels");
    const size_t numOutputClasses = labelsStreamDesc.m_sampleLayout[0];

    const size_t numLSTMLayers = 3;
    const size_t cellDim = 1024;
    const size_t hiddenDim = 512;

    auto classifierOutputFunction = ResNet34ClassifierNet(inputImageShape, numOutputClasses);

    Variable labelsVar = Variable({ numOutputClasses }, L"Labels");

    auto trainingLossFunction = CNTK::CrossEntropyWithSoftmax(classifierOutputFunction, labelsVar, L"lossFunction");
    auto predictionFunction = CNTK::PredictionError(classifierOutputFunction, labelsVar, L"predictionError");

    auto imageClassifier = CNTK::Combined({ trainingLossFunction, predictionFunction }, L"ImageClassifier");

    size_t momentumTimeConstant = 1024;
    double learningRatePerSample = 0.05;

    // Train for 100000 samples; checkpoint every 10000 samples
    TrainingControlPtr driver = CNTK::BasicTrainingControl(100000, 10000, { L"LSTMClassifier.net", L"LSTMClassifier.ckp" });
    Trainer imageClassifierTrainer(imageClassifier, trainingLossFunction, { CNTK::SGDLearner(imageClassifier->Parameters(), learningRatePerSample, momentumTimeConstant) });

    std::unordered_map<Variable, StreamDescription> modelArgumentToReaderStreamMap = { { classifierOutputFunction->Argument(), imagesStreamDesc }, { labelsVar, labelsStreamDesc } };
    imageClassifierTrainer.Train(imageReader, modelArgumentToReaderStreamMap, driver);
}
