#pragma once

#include "NDArrayView.h"

namespace CNTK
{
    // The Value type denotes a multi-dimensional array of values with an optional mask
    // This denotes the actual data fed into or produced from a computation
    class Value
    {
    public:
        // An empty Value
        Value(DeviceDescriptor device = DeviceDescriptor::DefaultDevice());

        // A multi-dimensional value with no mask
        Value(const NDArrayView& data); 

        // The mask allows specifying certain locations in data array to be marked as invalid for purposes of batching variable length sequences.
        // The mask array view is typically lower dimensionailty than the data, which means values are masked in units of (data.rank() - mask.rank()) 
        // dimensional values along the least significat dimensions of the data
        // Note: The data and mask must be on the same device
        Value(const NDArrayView& data, const NDArrayView& mask);

        NDArrayView Data() const;
        NDArrayView Mask() const;

        DeviceDescriptor Device() const;

        Value DeepClone(bool readOnly = false) const;
    };

    // Builtin methods
    NDArrayView RandomNormal(const NDShape& shape, double mean, double stdDev, DeviceDescriptor device = DeviceDescriptor::DefaultDevice());
    NDArrayView RandomUniform(const NDShape& shape, double rangeStart, double rangeEnd, DeviceDescriptor device = DeviceDescriptor::DefaultDevice());
    NDArrayView Constant(const NDShape& shape, double value, DeviceDescriptor device = DeviceDescriptor::DefaultDevice());
}
