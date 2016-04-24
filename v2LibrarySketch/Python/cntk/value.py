#
# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
#

from ndarray_view import *

class Value:

    # The Value type denotes a multi-dimensional array with an optional mask
    # This denotes the actual data fed into or produced from a computation

    def __init__(self, data: NDArrayView, mask: NDArrayView = None):
        # The mask allows specifying certain locations in data array to be marked as invalid for purposes of batching variable length sequences.
        # Only contiguous ranges at the end the end of axes can be masked/invalid.
        # The mask array view is typically lower dimensionailty than the data, which means values are masked in units of (data.rank() - mask.rank()) 
        # dimensional values along the least significat dimensions of the data
        # Note: The data and mask must be on the same device
        pass

    def data(self) -> NDArrayView:
        pass

    def mask(self) -> NDArrayView:
        pass

    def device(self) -> DeviceDescriptor:
        pass

    def deep_clone(self, readOnly: bool = False) -> Value:
        pass

# Built-in methods for constructing NDArrayView objects
def random_normal(shape: NDShape, mean: float, stdDev: float, device: DeviceDescriptor = DeviceDescriptor.default_device()) -> NDArrayView:
        pass

def random_uniform(shape: NDShape, rangeStart: float, rangeEnd: float, device: DeviceDescriptor = DeviceDescriptor.default_device()) -> NDArrayView:
        pass

def constant(shape: NDShape, value: float, device: DeviceDescriptor = DeviceDescriptor.default_device()) -> NDArrayView:
        pass
