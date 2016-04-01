#pragma once

#include <vector>
#include <memory>

namespace CNTK
{
    enum class ValueType
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
        Complex,
    };

    enum class StorageType
    {
        DENSE,
        SPARSE_CSC,
        // TODO: Others?
    };

    enum class DeviceType
    {
        CPU,
        GPU,
        FPGA,
    };

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

    typedef std::vector<size_t> NDShape;
    const size_t INFERRED_DIMENSION = -1;

    // Represents a multi-dimensional array of values.
    // This type denotes a view and there maybe multiple simultaneous views of the data underlying a NDArrayView instance.
    // The underlying data may be stored in sparse or dense form, and is located on the CPU or the GPU. 
    // The actual storage is either external or internal in which case its lifetime is managed through reference counting
    // The view may be writable or read-only
    class NDArrayView
    {
    public:

    public:
        // Construct a N dimensional view over a dense CPU buffer
        NDArrayView(void* buffer, size_t bufferSizeInBytes, ValueType dataType, const NDShape& viewShape, bool readOnly = false);

        // TODO: Define the full set of constructors for creating views over sparse as well as dense buffers on the CPU or a GPU.

        DeviceDescriptor Device() const;
        ValueType ElementType() const;
        StorageType StorageType() const;

        // TODO: Methods to access the raw storage underlying the view

        const NDShape& Shape() const;

        // Performs a deep copy of the view's contents and returns a view over the copied data
        NDArrayView DeepClone(bool readOnly = false) const;
    };
}
