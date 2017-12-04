//
// Copyright (c) Microsoft. All rights reserved.
//
// Licensed under custom Microsoft Research License Terms for
// 1-bit Stochastic Gradient Descent.
// See LICENSE.md file in the project root for full license information.
//

#pragma  once

#include <vector>
#include "CNTKLibrary.h"
#include "DistributedLearnerBase.h"
#include "PerformanceProfiler.h"

namespace CNTK
{
    ///
    /// Quantized Distributed Trainer.
    ///
    class QuantizedDataParallelDistributedLearner : public DistributedLearnerBase
    {
    public:
        QuantizedDataParallelDistributedLearner(QuantizedDistributedCommunicatorPtr communicator, LearnerPtr learner, size_t distributeAfterSamples, bool useAsyncBufferedParameterUpdate)
            : DistributedLearnerBase(communicator, learner, distributeAfterSamples, /*convertSparseToDense=*/false)
        {
            if (useAsyncBufferedParameterUpdate)
                LogicError("Asynchronous parameter update is not yet supported.");
        }

        // Optional override that gets called per minibatch after finishing gradient computation but before updating model parameters
        bool Update(std::unordered_map<Parameter, NDArrayViewPtr>& gradientValues, MinibatchInfo& info) override
        {
            // sparse gradient may be converted to dense for aggregation
            std::unordered_map<Parameter, NDArrayViewPtr> convertedGradientValues = gradientValues;

            if (m_sampleCount >= m_distributeAfterSamples &&
                (m_communicator->Workers().size() > 1 || MPICommunicatorImpl::ALWAYS_COMMUNICATE))
                //if (m_sampleCount >= m_distributeAfterSamples)
            {
                auto profGradientAgg = Microsoft::MSR::CNTK::ScopeProfile(Microsoft::MSR::CNTK::profilerEvtMainGradient);

                if (info.IsEmpty())
                    PrepaireZeroGradients(gradientValues, info);

                ConvertToOrdered(gradientValues, m_gradientBuffer, &convertedGradientValues);

                // predicate to define what is a small object
                // Small objects are meant to bypass quantization for biases and Droppo stabilizers,
                // which are small but have a large impact on the objective and should therefor be accurate.
                const auto isSmallObject = [](const NDArrayViewPtr& view)
                {
                    const auto& shape = view->Shape();
                    return shape.Rank() == 1 || shape.TotalSize() <= 1024; // 1024 is arbitrary; I want to catch tiny matrices
                };

                // TODO: This code was duplicated and modified from DataParallelDistributedLearner.cpp
                std::vector<NDArrayViewPtr> sparseValuesToAggregate;
                std::vector<NDArrayViewPtr> smallObjectsToAggregate;
                for (const auto& i : m_gradientBuffer)
                {
                    auto storageFormat = i.second->GetStorageFormat();
                    if (storageFormat == StorageFormat::SparseBlockCol)
                    {
                        // NOTE: CPU sparse block column stores block Ids in size_t and it's different from GPU SBC
                        // We should refactor the CPU SBC code to align with GPU in future
                        if (i.second->Device().Type() == DeviceKind::CPU)
                            LogicError("Unsupported CPU sparse block column aggregation");

                        sparseValuesToAggregate.push_back(i.second);
                    }
                    else if (isSmallObject(i.second))
                    {
                        smallObjectsToAggregate.push_back(i.second);
                    }
                }

                smallObjectsToAggregate.push_back(info.evalCriterionValue);
                smallObjectsToAggregate.push_back(info.trainingLossValue);

                auto value = MakeSharedObject<NDArrayView>(static_cast<double>(info.numberOfSamples), NDShape{ 1 }, DeviceDescriptor::CPUDevice());
                smallObjectsToAggregate.push_back(value);

                m_communicator->AggregateInPlace(smallObjectsToAggregate, m_communicator->Workers());

                info.numberOfSamples = static_cast<size_t>(*smallObjectsToAggregate.back()->DataBuffer<double>());

                std::vector<NDArrayViewPtr> gradients;
                for (const auto& i : m_gradientBuffer)
                {
                    auto storageFormat = i.second->GetStorageFormat();
                    if (storageFormat != StorageFormat::SparseBlockCol && !isSmallObject(i.second))
                        gradients.push_back(i.second);
                }
                m_gradientBuffer.clear();

                dynamic_cast<QuantizedDistributedCommunicator*>(m_communicator.get())->QuantizedAggregateInPlace(
                    gradients,
                    m_residuals,
                    m_stripeResiduals,
                    m_communicator->Workers());

                // sparse gradients are not quantized
                if (!sparseValuesToAggregate.empty())
                {
                    m_communicator->AllReduceSparseBlockColumn(sparseValuesToAggregate);
                }
            }

            auto profWeights = Microsoft::MSR::CNTK::ScopeProfile(Microsoft::MSR::CNTK::profilerEvtMainWeights);

            m_sampleCount += info.numberOfSamples;
            if (info.IsEmpty())
                return false;

            return m_learner->Update(convertedGradientValues, info.numberOfSamples, info.atEndOfSweep);
        }

        // Optionally overridable method to get checkpoint state associated with this Distributed train method
        Dictionary CreateCheckpoint() override
        {
            // Resetting the residuals.
            // We do this to make sure that the returned checkpoint state is consistent with the in - memory state, since we do not checkpoint the residues.
            for (size_t i = 0; i < m_residuals.size(); ++i)
                if (m_residuals[i]->GetDataType() == DataType::Double)
                    m_residuals[i]->SetValue(0.0);
                else
                    m_residuals[i]->SetValue(0.0f);

            for (size_t i = 0; i < m_stripeResiduals.size(); ++i)
                if (m_stripeResiduals[i])
                    if (m_stripeResiduals[i]->GetDataType() == DataType::Double)
                        m_stripeResiduals[i]->SetValue(0.0);
                    else
                        m_stripeResiduals[i]->SetValue(0.0f);

            return DistributedLearnerBase::CreateCheckpoint();
        }

    private:
        // Residuals of quantized gradients.
        std::vector<NDArrayViewPtr> m_residuals;
        // Residuals of quantized aggregated stripes this node is responsible for.
        std::vector<NDArrayViewPtr> m_stripeResiduals;
    };
}