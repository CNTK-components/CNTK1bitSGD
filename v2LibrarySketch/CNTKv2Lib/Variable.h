#pragma once

#include "Common.h"
#include <vector>
#include "Value.h"
#include <unordered_set>

namespace CNTK
{
    // Denotes an Axis of a Variable and is used for specifying the axes parameters of certain built-in fucntions such as reductions
    // Note that besides the axes corresponding to each of the ranks of the Variable's shape, a Variable (except for Parameter and Constants)
    // also has zero or more implicit sequence axes (corresponding to the sequence dimensions) and one implicit batch axes corresponding 
    // to the batching of multiple samples in input Values processed by a Function
    class AxisId
    {
    public:
        AxisId(size_t rankId);
        static AxisId DefaultSequenceAxis;
        static AxisId BatchAxisId;
        static AxisId NewAxis(const std::wstring& name = L"");

        std::wstring Name();
    };

    enum class VariableType
    {
        Constant,
        Parameter,
        Input,
        Output
    };

    // Variable instances are symbolic entities representing the inputs and outputs of a Function
    // Note that a Variable is symbolic and does not represent the actual values except when the 
    // Variable is a Constant or a Parameter in which case there is an actual Value bound to the variable
    // TODO: Should Parameter and Constant be separate types derived from Variable?
    class Variable
    {
    public:
        // Create an 'Input' Variable
        Variable(const NDShape& shape, const std::wstring& name = L"");
        Variable(const NDShape& shape, DataType type, const std::wstring& name = L"");

        // Create an 'Input' Variable with an explicitly specified sequenceAxisId
        // TODO: Do we need the ability to create Input variables with multiple sequenceAxes?
        Variable(const NDShape& shape, AxisId axisId = AxisId::DefaultSequenceAxis, DataType type = DataType::Float, const std::wstring& name = L"");

        // Create an 'Output' variable aliasing the Output of the specified Function
        Variable(FunctionPtr function);

        // Create a 'Constant' or 'Parameter' variable
        Variable(const NDArrayView& value, VariableType type = VariableType::Constant, const std::wstring& name = L"");

        // Create a 'Constant' variable denoting a scalar value
        template <typename T>
        Variable(T scalarValue);

        NDShape Shape() const;
        VariableType Type() const;
        std::wstring Name() const;

        std::unordered_set<AxisId> SequenceAxes() const;

        // Returns the value associated with a 'Constant' Variable
        // Throws an exception when called for a Variable that is not a 'Constant'
        // The returned NDArrayView is read-only and its contents cannot be mutated
        NDArrayView ConstantValue() const;

        // Returns the value associated with a 'Parameter' Variable
        // Throws an exception when called for a Variable that is not a 'Parameter'
        NDArrayView ParameterValue() const;

        // Function whose output this variable is. Only applicable for 'Output' variables 
        // Returns null when called for a Variable that is not an 'Output'
        FunctionPtr Owner() const;

        DataType DataType() const;

        Variable Clone(const std::wstring& name = L"") const;
    };

    bool operator==(const Variable& first, const Variable& second);

    // Built-in functions
    Variable Constant(const NDArrayView& value, const std::wstring& name = L"");
    Variable Parameter(const NDArrayView& value, const std::wstring& name = L"");
}

namespace std {
    template <> struct hash<CNTK::Variable>
    {
        size_t operator()(const CNTK::Variable& x) const;
    };

    template <> struct hash<CNTK::AxisId>
    {
        size_t operator()(const CNTK::AxisId& x) const;
    };
}


