#pragma once

#include <vector>
#include <memory>

namespace CNTK
{
    enum class DataType
    {
        Bit,
        Char,
        UChar,
        Short,
        UShort,
        Int,
        UInt,
        Long,
        ULong,
        Float8,
        Float16,
        Float,
        Double,
        // TODO: Complex, String ?
    };

    enum class StorageType
    {
        Dense,
        SParseCSC,
        // TODO: Others?
    };

    enum class DeviceType
    {
        CPU,
        GPU,
        FPGA,
    };

    // Descriptor for a specific compute deivce
    class DeviceDescriptor
    {
    public:
        int Id() const;
        DeviceType Type() const;

        static std::unordered_set<DeviceDescriptor> AllDevices();
        static DeviceDescriptor DefaultDevice();

        // The default device can only be changed if it has not yet been implicitly used by any previous operation in the CNTK library.
        static void SetDefaultDevice(DeviceDescriptor newDefault);

        static DeviceDescriptor BestDevice();
    };

    class NDShape : public std::vector<size_t>
    {
    public:
        NDShape(const std::initializer_list<size_t>& shapeDims);

        // Create a new NDShape that is a concatenation of 'this' and passed 'shape' appended to it
        NDShape AppendShape(const NDShape& shape) const;

        // TODO: Other methods
    };

    const size_t INFERRED_DIMENSION = -1;
    const size_t BATCH_AXIS = -10000;

    // Represents a multi-dimensional array of values.
    // This type denotes a view and there maybe multiple simultaneous views of the data underlying a NDArrayView instance.
    // The underlying data may be stored in sparse or dense form, and is located on the CPU or one of the GPU devices. 
    // The actual storage is either external or internal in which case its lifetime is managed through reference counting
    // The view may be writable or read-only
    class NDArrayView
    {
    public:

    public:
        // Construct a N dimensional view over a dense CPU buffer
        NDArrayView(void* buffer, size_t bufferSizeInBytes, DataType dataType, const NDShape& viewShape, bool readOnly = false);

        NDArrayView(const NDShape& shape, DataType dataType, DeviceDescriptor device = DeviceDescriptor::DefaultDevice());

        // An empty NDArrayView
        NDArrayView(DeviceDescriptor device = DeviceDescriptor::DefaultDevice());

        // A NDArrayView representing a scalar value
        template <typename T>
        NDArrayView(T scalarValue, DeviceDescriptor device = DeviceDescriptor::DefaultDevice());

        // TODO: Define the full set of constructors for creating views over sparse as well as dense buffers on the CPU or a GPU.

        DeviceDescriptor Device() const;
        DataType DataType() const;
        StorageType StorageType() const;

        // TODO: Methods to access the raw storage underlying the view

        const NDShape& Shape() const;

        // Performs a deep copy of the view's contents and returns a view over the copied data
        NDArrayView DeepClone(bool readOnly = false) const;

        template <typename ElemType>
        void SetValue(ElemType value) const;

        // The source must be of the same shape as 'this' NDArrayView
        void CopyFrom(NDArrayView source) const;

        NDArrayView Slice(size_t axis, size_t startIdx, size_t endIdx);
    };
}
