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

namespace CNTK
{
    ///
    /// Quantized Distributed Trainer.
    ///
    class QuantizedDataParallelDistributedLearner : public DistributedLearnerBase
    {
    public:
        QuantizedDataParallelDistributedLearner(QuantizedDistributedCommunicatorPtr communicator, bool useAsyncBufferedParameterUpdate, const std::vector<LearnerPtr>& learners)
            : DistributedLearnerBase(communicator, learners)
        {
            if (useAsyncBufferedParameterUpdate)
                LogicError("Asynchronous parameter update is not yet supported.");
        }

        // Optional override that gets called per minibatch after finishing gradient computation but before updating model parameters
        bool Update(std::vector<std::pair<Parameter, NDArrayViewPtr>>& gradientValues, MinibatchInfo& info, size_t& totalNumberOfSampleSeen) override
        {
            if (info.IsEmpty())
                PrepaireZeroGradients(gradientValues, info);

            std::vector<NDArrayViewPtr> headerToAggregate;
            headerToAggregate.push_back(info.evalCriterionValue);
            headerToAggregate.push_back(info.trainingLossValue);

            auto value = MakeSharedObject<NDArrayView>(static_cast<double>(info.numberOfSamples), NDShape{ 1 }, DeviceDescriptor::CPUDevice());
            headerToAggregate.push_back(value);

            m_communicator->AggregateInPlace(headerToAggregate, m_communicator->Workers());

            info.numberOfSamples = static_cast<size_t>(*headerToAggregate.back()->DataBuffer<double>());

            std::vector<NDArrayViewPtr> gradients;
            for (const auto& i : gradientValues)
                gradients.push_back(i.second);

            dynamic_cast<QuantizedDistributedCommunicator*>(m_communicator.get())->QuantizedAggregateInPlace(
                gradients,
                m_residuals,
                m_stripeResiduals,
                m_communicator->Workers());

            totalNumberOfSampleSeen += info.numberOfSamples;
            m_learner->Update(gradientValues, info, totalNumberOfSampleSeen);

            return info.IsEmpty();
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

            return Dictionary();
        }

    private:
        // Residuals of quantized gradients.
        std::vector<NDArrayViewPtr> m_residuals;
        // Residuals of quantized aggregated stripes this node is responsible for.
        std::vector<NDArrayViewPtr> m_stripeResiduals;
    };
}