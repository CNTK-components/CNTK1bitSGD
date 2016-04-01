#pragma once

#include <memory>
#include <assert.h>

#define DISALLOW_COPY_CTOR_AND_ASSIGNMENT(TypeName) \
    TypeName(const TypeName&) = delete; \
    TypeName& operator=(const TypeName&) = delete;

#define DISALLOW_MOVE_CTOR_AND_ASSIGNMENT(TypeName) \
    TypeName(TypeName&&) = delete; \
    TypeName& operator=(TypeName&&) = delete;

namespace CNTK
{
    // Forward declarations
    class Function;
    typedef std::shared_ptr<Function> FunctionPtr;

    class Trainer;
}
