#pragma once

#include "Function.h"
#include <unordered_map>

namespace CNTK
{
    // An object that encapsulates a Function and Values for the subset of the inputs of the Function bound as parameter inputs
    // The inputs of the Model are inputs of the specified rootFunction minus variables bound as parameters
    class Model : public Function
    {
    public:
        Model(const FunctionPtr& rootFunction, const std::unordered_map<Variable, const ValuePtr>& parameterValues);
        Model(const std::wstring& modelPath);

        virtual void Evaluate(const std::unordered_map<Variable, const ValuePtr>& inputs,
                              const std::unordered_map<Variable, ValuePtr>& outputs);

        virtual BackPropStatePtr EvaluateForBackProp(const std::unordered_map<Variable, const ValuePtr>& inputs,
                                                     const std::unordered_map<Variable, ValuePtr>& outputs);

        virtual void BackPropagate(BackPropStatePtr state,
                                   const std::unordered_map<Variable, const ValuePtr> rootGradientValues,
                                   const std::unordered_map<Variable, ValuePtr>& backPropagatedGradientValuesForInputs);

        void Save(const std::wstring& modelPath);

        FunctionPtr RootFunction();
        const std::unordered_map<Variable, const ValuePtr>& Parameters();
    };
}
