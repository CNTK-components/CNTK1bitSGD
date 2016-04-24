#pragma once

#include "Common.h"
#include <set>
#include <unordered_set>
#include <unordered_map>
#include "Variable.h"
#include "Value.h"

namespace CNTK
{
    // An opaque type encapsulating the internal state of a network used for passing intermediate values
    // across from the 'Forward' call on a Function to a later 'Backward' call on the same Function for the same
    // computation corresponding to the 'Forward' call
    class BackPropState 
    {
    public:
        Function Function() const;
        DeviceDescriptor Device() const;

        BackPropState() = delete;
    };

    // A type denoting a dictionary (keyed by Unicode strings) of serializable values (dynamically typed). A serializable value represents one of:
    // a) Boolean
    // b) String
    // c) Double precision floating point value
    // d) Signed long integer
    // e) std::vector<SerializationValue>
    // f) Dictionary
    class Dictionary;
    
    // Represents a function (optionally differentiable)
    // A function is a symbolic entity with zero or more input arguments and one or more outputs. 
    // A function may be primitive or composite (comprised of other function instances whose inputs and outputs are wired together in some form)
    // Note that above definition is recursive and effectively means that a function is an arbitrary graph composed of other primitive functions
    class Function
    {
    public:
        // The methods in the following section are the only methods that Function's derived types must implement.

        // Computes and stores the values of speficied variables in the 'outputs' map, using provided 'inputs' values corresponding to each input variable of the function.
        // The variables specified in the 'outputs' map denote the set of output or intermediate computed variables that the caller wants to obtain 'Value's of. 
        // Note that callers may choose to explicitly specify the actual storage to be used for storing the 'outputs' Values or leave them to be null in which case
        // the implementation allocates the actual storage for the 'outputs' for which the ValuePtr mapping was left null by the caller
        // The optional 'returnStateForBackward' specifies whether the method should return a BackPropState object containing all intermediate variable values
        // that may be needed during backpropagation of gradients from the outputs of the function to any of the inputs of the Function, in a subsequent Backward call.
        // Note that the returned BackPropState instance also stores a reference to the supplied 'inputs' Values and generated 'outputs' Values
        // and the user is responsible for ensuring that the contents of the inputs and outputs are unchanged until after any uses of the BackPropState instance
        // for backpropagating gradients through this function
        virtual BackPropState Forward(const std::unordered_map<Variable, const Value>& arguments,
                                      const std::unordered_map<Variable, Value>& outputs,
                                      DeviceDescriptor computeDevice = DeviceDescriptor::DefaultDevice(),
                                      bool returnStateForBackward = false) = 0;

        // Back propagates supplied 'rootGradientValues' for one or more of the output variables of the function, through the function to produce gradient Values
        // corresponding to the specified set of input variables in 'backPropagatedGradientValuesForInputs'.
        // Note that callers may choose to explicitly specify the actual storage to be used for storing the 'backPropagatedGradientValuesForInputs' Values or leave them
        // to be null in which case the implementation allocates the actual storage for the 'backPropagatedGradientValuesForInputs' for which the ValuePtr mapping was left null by the caller
        // In case an existing storage is specified, the gradients are aggregated with existing values in the storage instead of being overwritten
        // The 'state' parameter is an instance of an BackPropState instance obtained from a previous call to the EvaluateForBackProp method on this Function instance for the 
        // computation that this gradient backpropagation corresponds to.
        virtual void Backward(BackPropState state,
                              const std::unordered_map<Variable, Value> rootGradientValues,
                              const std::unordered_map<Variable, Value>& backPropagatedGradientValuesForInputs) = 0;

    public:
        // Optionally overridable methods

        // An optionally overridable method that specifies which inputs does this function backpropagate gradients to. 
        // By default a function is assumed to backpropgate gradients to all of its inputs
        virtual std::unordered_set<Variable> InputsBackPropagatedTo();

        // An optionally overridabe method that specifies, Values of which of the Function's ouptuts are needed by the function to backpropagate gradients
        // from its outputs to its inputs
        virtual std::unordered_set<Variable> OutputsRequiredForBackProp();

        // An optionally overridabe method that specifies, Values of which of the Function's inputs are needed by the function to backpropagate gradients
        // from its outputs to its inputs
        virtual std::unordered_set<Variable> InputsRequiredForBackProp();

        // An optional method that primitive Function types deriving from this abstract type can implement to generate
        // a dictionary containing the serialization parameters to be needed for reconstructing an instance of the function
        virtual Dictionary Serialize();

        // This method must be implemented by all derivatives of the Function type which want to be serialized/deserialized as part of the Function serialization and deserialization capablity
        // This method must return a pair consisting of the name of the DLL/shared-library and the name of the method in the DLL/shared-library that is to be used for contructing an new instance of the Function.
        // This construction method must have the following signature:
        // FunctionPtr (*)(const std::unordered_set<Variable>& inputs, const std::unordered_set<Variable>& outputs, const Dictionary& serializedData)
        virtual std::pair<std::wstring, std::wstring> FunctionInstanceCreatorMethodInfo();

        // Creates a clone of the function instance
        // TODO: Should this be a mandatory override
        virtual FunctionPtr Clone() const;

    public:
        // Conversion operator that returns the first Output variable of the function; i.e. Outputs()[0]
        // TODO: Throw an exception when called for a function with multiple outputs?
        operator Variable() const;

        std::wstring Name() const;

        // Returns the Function at the root of the graph of Functions underlying this Function
        // If this is a primitive function then (this->RootFunction() == this)
        FunctionPtr RootFunction() const;

        // All Output variables
        std::vector<Variable> Outputs() const;

        // All input variables (inlcudes 'Parameter' and 'Constant' inputs)
        std::vector<Variable> Inputs() const;

        // All input variables that are not 'Parameter's or 'Constant's and constitute the Arguments of the function
        std::vector<Variable> Arguments() const;

        // All Parameter variables
        std::unordered_set<Variable> Parameters() const;

        // All Parameter values
        std::unordered_map<Variable, Value> ParametersValues() const;

        // Value of a specific Parameter
        Value ParameterValue(Variable param) const;

        // All Constant variables
        std::unordered_set<Variable> Constants() const;

        // All Constant values
        std::unordered_map<Variable, Value> ConstantsValues() const;

        // TODO: Methods to reflect on the Function's underlying graph structure
        // Provide the ability to "Visit" the graph, which can be used to achieve model editing functionality 

    protected:
        // Protected ctor for derived 'Function' ctors to specify the actual input parameters and shapes of outputs for a function instance.
        // All 'inputs; specified must be Variables of type Constant, Parameter or Input
        Function(const std::vector<Variable>& inputs, const std::vector<NDShape>& outputShapes, const std::wstring& name = L"");
    };

    // Factory methods to instantiate built-in CNTK functions
    FunctionPtr Times(const Variable& leftOperand, const Variable& rightOperand, const std::wstring& name = L"");
    FunctionPtr Plus(const Variable& leftOperand, const Variable& rightOperand, const std::wstring& name = L"");
    FunctionPtr ReLU(const Variable& operand, const std::wstring& name = L"");
    FunctionPtr Sigmoid(const Variable& operand, const std::wstring& name = L"");
    FunctionPtr Tanh(const Variable& operand, const std::wstring& name = L"");
    FunctionPtr CrossEntropyWithSoftmax(const Variable& output, const Variable& labels, const std::wstring& name = L"");
    FunctionPtr PredictionError(const Variable& prediction, const Variable& labels, const std::wstring& name = L"");
    FunctionPtr Exp(const Variable& operand, const std::wstring& name = L"");
    FunctionPtr PastValue(const Variable& initialState, const Variable& operand, const std::wstring& name = L"");
    FunctionPtr PastValue(const Variable& initialState, const Variable& operand, AxisId axis, const std::wstring& name = L"");
    FunctionPtr ElementTimes(const Variable& leftOperand, const Variable& rightOperand, const std::wstring& name = L"");
    FunctionPtr Convolution(const Variable& convolutionMap, const Variable& operand, const NDShape& strides, bool zeroPadding = false, const std::wstring& name = L"");
    FunctionPtr BatchNormalization(const Variable& operand, const Variable& scale, const Variable& bias, const Variable& runningMean, const Variable& runningInvStd, bool spacial, size_t bnTimeConstant, double epsilon, const std::wstring& name = L"");

    FunctionPtr IsLess(const Variable& leftOperand, const Variable& rightOperand, const std::wstring& name = L"");
    FunctionPtr IsGreater(const Variable& leftOperand, const Variable& rightOperand, const std::wstring& name = L"");
    FunctionPtr IsEqual(const Variable& leftOperand, const Variable& rightOperand, const std::wstring& name = L"");
    FunctionPtr LogicalNot(const Variable& operand, const std::wstring& name = L"");
    FunctionPtr Conditional(const Variable& predicate, const Variable& trueConditionOperand, const Variable& falseConditionOperand, const std::wstring& name = L"");

    // Operator overloads
    FunctionPtr operator+(const Variable& leftOperand, const Variable& rightOperand);
    FunctionPtr operator*(const Variable& leftOperand, const Variable& rightOperand);

    enum class PoolingType
    {
        Max,
        Average,
    };

    FunctionPtr Pooling(const Variable& operand, PoolingType poolingType, const NDShape& poolingWindowShape, const NDShape& strides, const std::vector<bool>& autoPadding = {} /* defaults to no padding */, const std::wstring& name = L"");

    FunctionPtr Softmax(const Variable& operand);

    // Overload to perform SoftMax reduction along the specified axis
    FunctionPtr Softmax(const Variable& operand, AxisId axis);

    FunctionPtr Reshape(const Variable& operand, size_t beginAxis, size_t endAxis, const NDShape& newShape);

    FunctionPtr Gather(const Variable& gatherFrom, const Variable& gatherIndices, const std::wstring& name = L"");

    FunctionPtr RowStack(const Variable& top, const Variable& bottom, const std::wstring& name = L"");

    FunctionPtr Sum(const Variable& operand, AxisId reductionAxis = 0, const std::wstring& name = L"");
    FunctionPtr Average(const Variable& operand, AxisId reductionAxis = 0, const std::wstring& name = L"");

    // Method to create a Composite function whose root is a clone of the specified 'rootFunction' and the inputs of the root function are wired to the specified
    // 'rootFunctionInputsConnections' map to effectively compose a graph. Note that specified rootFunctionInputsConnections may be outputs of other Functions.
    // Note that this does not modify the supplied 'rootFunction'
    FunctionPtr Composite(FunctionPtr rootFunction, const std::unordered_map<Variable, Variable>& rootFunctionInputsConnections, const std::wstring& name = L"");

    // Create a block function wrapping a specified function whose Output variables are distinct aliases of the specified root function 
    // such that the block retains its identity when composed with other functions
    // This is to enable creating a given composite function as a Block that appears as a primitive when traversing a graph of Functions.
    // For e.g.one could take a LSTM loop and create a block out of it and stack multiple of these blocks together.Now when traversing this structure,
    // this will appear as a feed forward network with 3 of these block functions chained.
    FunctionPtr Block(FunctionPtr rootFunction, const std::wstring& name);

    // Create a new combined function whose inputs and outputs are the union of the inputs of the specified set of rootFunctions
    // This can be used to combine multiple functions into a single function
    // E.g. The model for a classification problem comprises of a training loss function to use as the training objective with and an error prediciton function
    FunctionPtr Combined(std::unordered_set<FunctionPtr> rootFunctions, const std::wstring& name = L"");
}
