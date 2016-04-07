#include "Function.h"

namespace CNTK
{
    inline FunctionPtr Scale(const Variable& scaleFactor, const Variable& operand, const std::wstring& name = L"")
    {
        return CNTK::ElementTimes(scaleFactor, operand, name);
    }
}
