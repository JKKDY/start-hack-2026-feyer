#include "dfm/version.hpp"

#ifndef DFM_VERSION_STRING
#define DFM_VERSION_STRING "unknown"
#endif

namespace dfm {

const char* version() noexcept { return DFM_VERSION_STRING; }

}  // namespace dfm
