#
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
#

from variable import *
from value import *

class BackPropState:

    # An opaque type encapsulating the internal state of a network used for passing intermediate values
    # across from the 'Forward' call on a Function to a later 'Backward' call on the same Function for the same
    # computation corresponding to the 'Forward' call

    def function(self) -> Function:
        pass

    def device(self) -> DeviceDescriptor:
        pass

class Dictionary:

    # A type denoting a dictionary (keyed by Unicode strings) of serializable values (dynamically typed). A serializable value represents one of:
    # a) Boolean
    # b) String
    # c) Double precision floating point value
    # d) Signed long integer
    # e) std::vector<SerializationValue>
    # f) Dictionary
    pass

class Function:
    __metaclass__ = ABCMeta

    # Represents a function (optionally differentiable)
    # A function is a symbolic entity with zero or more input arguments and one or more outputs. 
    # A function may be primitive or composite (comprised of other function instances whose inputs and outputs are wired together in some form)
    # Note that above definition is recursive and effectively means that a function is an arbitrary graph composed of other primitive functions

    # The methods in the following section are the only methods that Function's derived types must implement.

    def forward(self,
                arguments: Dict[Variable, Value],
                outputs : Dict[Variable, Value],
                computeDevice: DeviceDescriptor = DeviceDescriptor.default_device(),
                returnStateForBackward: bool = False) -> BackPropState:

        # Computes and stores the values of speficied variables in the 'outputs' map, using provided 'inputs' values corresponding to each input variable of the function.
        # The variables specified in the 'outputs' map denote the Set of output or intermediate computed variables that the caller wants to obtain 'Value's of. 
        # Note that callers may choose to explicitly specify the actual storage to be used for storing the 'outputs' Values or leave them to be null in which case
        # the implementation allocates the actual storage for the 'outputs' for which the ValuePtr mapping was left null by the caller
        # The optional 'returnStateForBackward' specifies whether the method should return a BackPropState object containing all intermediate variable values
        # that may be needed during backpropagation of gradients from the outputs of the function to any of the inputs of the Function, in a subsequent Backward call.
        # Note that the returned BackPropState instance also stores a reference to the supplied 'inputs' Values and generated 'outputs' Values
        # and the user is responsible for ensuring that the contents of the inputs and outputs are unchanged until after any uses of the BackPropState instance
        # for backpropagating gradients through this function
        raise NotImplementedError()

    def backward(self,
                 state: BackPropState,
                 rootGradientValues: Dict[Variable, Value],
                 backPropagatedGradientValuesForInputs: Dict[Variable, Value]):

        # Back propagates supplied 'rootGradientValues' for one or more of the output variables of the function, through the function to produce gradient Values
        # corresponding to the specified Set of input variables in 'backPropagatedGradientValuesForInputs'.
        # Note that callers may choose to explicitly specify the actual storage to be used for storing the 'backPropagatedGradientValuesForInputs' Values or leave them
        # to be null in which case the implementation allocates the actual storage for the 'backPropagatedGradientValuesForInputs' for which the ValuePtr mapping was left null by the caller
        # In case an existing storage is specified, the gradients are aggregated with existing values in the storage instead of being overwritten
        # The 'state' parameter is an instance of an BackPropState instance obtained from a previous call to the EvaluateForBackProp method on this Function instance for the 
        # computation that this gradient backpropagation corresponds to.
        raise NotImplementedError()

    # Optionally overridable methods

    def inputs_back_propagated_to(self) -> Set[Variable]:
        # An optionally overridable method that specifies which inputs does this function backpropagate gradients to. 
        # By default a function is assumed to backpropgate gradients to all of its inputs
        pass

    def outputs_required_for_backprop(self) -> Set[Variable]:
        # An optionally overridabe method that specifies, Values of which of the Function's ouptuts are needed by the function to backpropagate gradients
        # from its outputs to its inputs
        pass

    def inputs_required_for_backprop(self) -> Set[Variable]:
        # An optionally overridabe method that specifies, Values of which of the Function's inputs are needed by the function to backpropagate gradients
        # from its outputs to its inputs
        pass

    def serialize(self) -> Dictionary:
        # An optional method that primitive Function types deriving from this abstract type can implement to generate
        # a dictionary containing the serialization parameters to be needed for reconstructing an instance of the function
        pass

    def function_instance_creator_method_info(self) -> Tuple[str, str]:
        # This method must be implemented by all derivatives of the Function type which want to be serialized/deserialized as part of the Function serialization and deserialization capablity
        # This method must return a pair consisting of the name of the DLL/shared-library and the name of the method in the DLL/shared-library that is to be used for contructing an new instance of the Function.
        # This construction method must have the following signature:
        # FunctionPtr (*)(const std::unordered_set<Variable>& inputs, const std::unordered_set<Variable>& outputs, const Dictionary& serializedData)
        pass

    def clone(self) -> Function:
        # Creates a clone of the function instance
        # TODO: Should this be a mandatory override
        pass

    def name(self) ->str:
        pass

    def root_function(self) -> Function:
        # Returns the Function at the root of the graph of Functions underlying this Function
        # If this is a primitive function then (this->RootFunction() == this)
        pass

    def outputs(self) -> List[Variable]:
        # All Output variables
        pass

    def inputs(self) ->List[Variable]:
        # All input variables (inlcudes 'Parameter' and 'Constant' inputs)
        pass

    def arguments(self) -> List[Variable]:
        # All input variables that are not 'Parameter's or 'Constant's and constitute the Arguments of the function
        pass

    def parameters(self) -> Set[Variable]:
        # All Parameter variables
        pass

    def parameters_values(self) -> Dict[Variable, Value]:
        # All Parameter values
        pass

    def parameter_value(self, param: Variable) -> Value:
        # Value of a specific Parameter
        pass

    def constants(self) -> Set[Variable]:
        # All Constant variables
        pass

    def constants_values(self) -> Dict[Variable, Value]:
        # All Constant values
        pass

    # TODO: Methods to reflect on the Function's underlying graph structure
    # Provide the ability to "Visit" the graph, which can be used to achieve model editing functionality 

    def __init__(self, inputs : List[Variable], outputShapes : List[NDShape], name: str = ""):
        # Protected ctor for derived 'Function' ctors to specify the actual input parameters and shapes of outputs for a function instance.
        # All 'inputs; specified must be Variables of type Constant, Parameter or Input
        pass

# Factory methods to instantiate built-in CNTK functions
def times(leftOperand : Variable, rightOperand : Variable, name: str = "") -> Function:
    pass

def plus(leftOperand : Variable, rightOperand : Variable, name: str = "") -> Function:
    pass

def relu(operand : Variable, name: str = "") -> Function:
    pass

def sigmoid(operand : Variable, name: str = "") -> Function:
    pass

def tanh(operand : Variable, name: str = "") -> Function:
    pass

def cross_entropy_with_softmax(output : Variable, labels : Variable, name: str = "") -> Function:
    pass

def prediction_error(prediction : Variable, labels : Variable, name: str = "") -> Function:
    pass

def exp(operand : Variable, name: str = "") -> Function:
    pass

def past_value(initialState : Variable, operand : Variable, axis : AxisId = AxisId.all_axes(), name: str = "") -> Function:
    pass

def element_times(leftOperand : Variable, rightOperand : Variable, name: str = "") -> Function:
    pass

def convolution(convolutionMap : Variable, operand : Variable, strides: NDShape, zeroPadding: bool = false, name: str = "") -> Function:
    pass

def batch_normalization(operand : Variable,
                        scale : Variable,
                        bias : Variable,
                        runningMean: Variable,
                        runningInvStd: Variable,
                        spacial: bool,
                        bnTimeConstant: int,
                        epsilon: float,
                        name: str = "") -> Function:
    pass

def is_less(leftOperand : Variable, rightOperand : Variable, name: str = "") -> Function:
    pass

def is_greater(leftOperand : Variable, rightOperand : Variable, name: str = "") -> Function:
    pass

def is_equal(leftOperand : Variable, rightOperand : Variable, name: str = "") -> Function:
    pass

def logical_not(operand : Variable, name: str = "") -> Function:
    pass

def conditional(predicate : Variable, trueConditionOperand : Variable, falseConditionOperand : Variable, name: str = "") -> Function:
    pass

class PoolingType(Enum):
    max,
    average

def pooling(operand : Variable,
            poolingType: PoolingType,
            poolingWindowShape: NDShape,
            strides: NDShape,
            autoPadding: List[bool] = {},
            name: str = "") -> Function:
    pass

def softmax(operand: Variable, axis: AxisId = 0) -> Function:
    # Overload to perform SoftMax reduction along the specified axis
    pass

def reshape(operand: Variable, beginAxis: int, endAxis: int, newShape: NDShape) -> Function:
    pass

def gather(gatherFrom: Variable, gatherIndices: Variable, name: str = "") -> Function:
    pass

def row_stack(top: Variable, bottom: Variable, name: str = "") -> Function:
    pass

def sum(operand: Variable, reductionAxis: AxisId = 0, name: str = "") -> Function:
    pass

def average(operand: Variable, reductionAxis: AxisId = 0, name: str = "") -> Function:
    pass

def composite(rootFunction: Function, rootFunctionInputsConnections: Dict[Variable, Variable], name: str = "") -> Function:
    # Method to create a Composite function whose root is a clone of the specified 'rootFunction' and the inputs of the root function are wired to the specified
    # 'rootFunctionInputsConnections' map to effectively compose a graph. Note that specified rootFunctionInputsConnections may be outputs of other Functions.
    # Note that this does not modify the supplied 'rootFunction'
    pass

def block(rootFunction: Function, name: str = "") -> Function:
    # Create a block function wrapping a specified function whose Output variables are distinct aliases of the specified root function 
    # such that the block retains its identity when composed with other functions
    # This is to enable creating a given composite function as a Block that appears as a primitive when traversing a graph of Functions.
    # For e.g.one could take a LSTM loop and create a block out of it and stack multiple of these blocks together.Now when traversing this structure,
    # this will appear as a feed forward network with 3 of these block functions chained.
    pass

def combined(rootFunctions: Set[Function], name: str = "") -> Function:
    # Create a new combined function whose inputs and outputs are the union of the inputs of the specified Set of rootFunctions
    # This can be used to combine multiple functions into a single function
    # E.g. The model for a classification problem comprises of a training loss function to use as the training objective with and an error prediciton function
    pass
