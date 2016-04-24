#
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
#

from typing import *

class DataType(Enum):
    bit,
    char,
    uchar,
    short,
    ushort,
    int,
    uint,
    long,
    ulong,
    float8,
    float16,
    float,
    double

class StorageType(Enum):
    dense,
    sparse_csc
    # TODO: Others?

class DeviceType(Enum):
    cpu,
    gpu,
    fpga

class DeviceDescriptor:

    # Descriptor for a specific compute deivce

    def id(self) -> int:
        pass

    def type(self) -> DeviceType:
        pass

    def all_devices() -> Set[DeviceDescriptor]:
        pass

    def default_device() -> DeviceDescriptor:
        pass

    def set_default_device(new_default: DeviceDescriptor):
        # The default device can only be changed if it has not yet been implicitly used by any previous operation in the CNTK library.
        pass

    def best_device() -> DeviceDescriptor:
        pass

class NDShape(List[int]):

    def __init__(self, shape_dims: List[int]):
        pass

    def append_shape(self, shape: NDShape) -> NDShape:
        # Create a new NDShape that is a concatenation of 'this' and passed 'shape' appended to it
        pass

    # TODO: Other methods

INFERRED_DIMENSION = -1
BATCH_AXIS = -10000

class NDArrayView:

    # Represents a multi-dimensional array of values.
    # This type denotes a view and there maybe multiple simultaneous views of the data underlying a NDArrayView instance.
    # The underlying data may be stored in sparse or dense form, and is located on the CPU or one of the GPU devices. 
    # The actual storage is either external or internal in which case its lifetime is managed through reference counting
    # The view may be writable or read-only

    def __init__(self, shape: NDShape, dataType: DataType, device: DeviceDescriptor = DeviceDescriptor.default_device()):
        pass

    # TODO: Define the full Set of constructors for creating views over sparse as well as dense buffers on the CPU or a GPU.

    def device(self) -> DeviceDescriptor:
        pass

    def data_type(self) -> DataType:
        pass

    def storage_type(self) -> StorageType:
        pass

    # TODO: Methods to access the raw storage underlying the view

    def shape(self) -> NDShape:
        pass

    def deep_clone(self, readOnly: bool = False) -> NDArrayView:
        # Performs a deep copy of the view's contents and returns a view over the copied data
        pass

    def set_value(self, value):
        pass

    def copy_from(self, source: NDArrayView):
        # The source must be of the same shape as 'this' NDArrayView
        pass

    def slice(self, axis: int, startIdx: int, endIdx: int) -> NDArrayView:
        pass
