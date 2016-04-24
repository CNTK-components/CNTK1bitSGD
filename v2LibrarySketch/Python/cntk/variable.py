#
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
#

from ndarray_view import *

class AxisId:

    # Denotes an Axis of a Variable and is used for specifying the axes parameters of certain built-in fucntions such as reductions
    # Note that besides the axes corresponding to each of the ranks of the Variable's shape, a Variable (except for Parameter and Constants)
    # also has zero or more implicit sequence axes (corresponding to the sequence dimensions) and one implicit batch axes corresponding 
    # to the batching of multiple samples in input Values processed by a Function

    def __init__(self, rankId: int):
        pass

    def default_dynamic_axis() -> AxisId:
        pass

    def batch_axis_id() -> AxisId:
        pass

    def all_axes() -> AxisId:
        pass

    def new_dynamic_axis(name: str = "") -> AxisId:
        pass

    def name(self) -> str:
        pass

class VariableType(Enum):
    constant,
    parameter,
    input,
    output

class Variable:

    # Variable instances are symbolic entities representing the inputs and outputs of a Function
    # Note that a Variable is symbolic and does not represent the actual values
    # Create an 'Input' Variable

    def __init__(self, shape: NDShape, type: DataType = DataType.float, axisId: AxisId = AxisId.default_dynamic_axis(), name: str = ""):
        # Create an 'Input' Variable with an explicitly specified dynamicAxisId
        # TODO: Do we need the ability to create Input variables with multiple dynamicAxes?
        pass

    def shape(self) -> NDShape:
        pass

    def type(self) -> VariableType:
        pass

    def name(self) -> str:
        pass

    def dynamic_axes(self) -> Set[AxisId]:
        pass

    def owner(self) -> Function:
        # Function whose output this variable is. Only applicable for 'Output' variables 
        # Returns null when called for a Variable that is not an 'Output'
        pass

    def data_type(self) -> DataType:
        pass

    def clone(self, name: str = "") -> Variable:
        pass

    # Operator overloads
    def __add__(self, other: Variable) -> Function:
        pass

    def __mul__(self, other: Variable) -> Function:
        pass

# Built-in methods for constructing constant and parameter Variable objects
def constant(value: NDArrayView, name: str = "") -> Variable:
    pass

def parameter(value: NDArrayView, name: str = "") -> Variable:
    pass
