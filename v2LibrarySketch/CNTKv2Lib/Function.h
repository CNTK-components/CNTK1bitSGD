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
    // A function is a symbolic entity with zero or more input parameters and one or more outputs. 
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
                              const std::unordered_map<Variable, const Value> rootGradientValues,
                              const std::unordered_map<Variable, Value>& backPropagatedGradientValuesForInputs) = 0;

    public:
        // Optionally overridable methods

        // An optionally overridable method that specifies to which of the Function's inputs are gradient values from outputs, backpropagted to. 
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
        // First Output variable
        Variable Output() const;

        // First Argument variable (i.e. input that is neither a Parameter nor a Constant)
        Variable Argument() const;

        // First Parameter Variable
        Variable Parameter() const;

        // Conversion operator that returns the first Output variable of the function, to be used as convenience for building composite functions
        operator Variable() const
        {
            return Output();
        }

        std::wstring Name() const;

        // Returns the Function at the root of the graph of Functions underlying this Function
        // If this is a primitive function (this->RootFunction() == this)
        FunctionPtr RootFunction() const;

        // All output variables whose names contain a substring matching the specified regular expression
        std::unordered_set<Variable> Outputs(const std::wstring& nameFilterRegex = L"");

        // All input vars (leaf descendants) whose names contain a substring matching the specified regular expression
        std::unordered_set<Variable> Arguments(const std::wstring& nameFilterRegex = L"");

        // All parameter vars (leaf descendants) whose names contain a substring matching the specified regular expression
        std::unordered_set<Variable> Parameters(const std::wstring& nameFilterRegex = L"");

        // TODO: Method to return 'Constant' input variables too

        // TODO: A bevy of helper methods to reflect on the Function's underlying graph structure
        // Provide the ability to "Visit" the graph, which can be used to achieve model editing functionality 

    protected:
        // Protected ctor for derived 'Function' ctors to specify the actual input parameters and shapes of outputs for a function instance.
        // All 'inputs; specified must be Variables of type Constant, Parameter or Input
        Function(const std::unordered_set<Variable>& inputs, const std::vector<NDShape>& outputShapes, const std::wstring& name = L"");

        DISALLOW_COPY_CTOR_AND_ASSIGNMENT(Function);
        DISALLOW_MOVE_CTOR_AND_ASSIGNMENT(Function);
    };

    // Built-in Functions
    FunctionPtr Times(const Variable& leftOperand, const Variable& rightOperand, const std::wstring& name = L"");
    FunctionPtr Plus(const Variable& leftOperand, const Variable& rightOperand, const std::wstring& name = L"");
    FunctionPtr ReLU(const Variable& operand, const std::wstring& name = L"");
    FunctionPtr Sigmoid(const Variable& operand, const std::wstring& name = L"");
    FunctionPtr Tanh(const Variable& operand, const std::wstring& name = L"");
    FunctionPtr CrossEntropyWithSoftmax(const Variable& output, const Variable& labels, const std::wstring& name = L"");
    FunctionPtr PredictionError(const Variable& prediction, const Variable& labels, const std::wstring& name = L"");
    FunctionPtr Exp(const Variable& operand, const std::wstring& name = L"");
    FunctionPtr PastValue(const Variable& initialState, const Variable& operand, const std::wstring& name = L"");
    FunctionPtr Scale(const Variable& scaleFactor, const Variable& operand, const std::wstring& name = L"");
    FunctionPtr DiagTimes(const Variable& leftOperand, const Variable& rightOperand, const std::wstring& name = L"");
    FunctionPtr ElementTimes(const Variable& leftOperand, const Variable& rightOperand, const std::wstring& name = L"");
    FunctionPtr Convolution(const Variable& convolutionMap, const Variable& operand, const NDShape& strides, bool zeroPadding = false, const std::wstring& name = L"");
    FunctionPtr BatchNormalization(const Variable& operand, const Variable& scale, const Variable& bias, const Variable& runningMean, const Variable& runningInvStd, bool spacial, size_t bnTimeConstant, double epsilon, const std::wstring& name = L"");

    enum class PoolingType
    {
        Max,
        Average,
    };
    FunctionPtr Pooling(const Variable& operand, PoolingType poolingType, const NDShape& poolingWindowShape, const NDShape& strides, const std::vector<bool>& autoPadding = {} /* defaults to no padding */, const std::wstring& name = L"");

    FunctionPtr Softmax(const Variable& operand, int axis = 0);
    FunctionPtr Reshape(const Variable& operand, int beginAxis, int endAxis, const NDShape& newShape);

    // Method to create a Composite function by wiring the inputs of a specified function with new Variables 
    // which in turn may be outputs of other Functions
    // Note that this does not modify the supplied 'rootFunction'
    FunctionPtr Composite(FunctionPtr rootFunction, const std::unordered_map<Variable, Variable>& rootFunctionInputsConnections, const std::wstring& name = L"");

    // Create a block function wrapping a specified function whose Output variables are distinct aliases of the specified root function 
    // such that the block retains its identity when composed with other functions
    FunctionPtr Block(FunctionPtr rootFunction, const std::wstring& name);

    // Create a new combined function whose inputs and outputs are the union of the inputs of the specified set of rootFunctions
    FunctionPtr Combined(std::unordered_set<FunctionPtr> rootFunctions, const std::wstring& name = L"");
}
