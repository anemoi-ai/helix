#include "log.hpp"

namespace helix {

Logger& Logger::instance() {
    static Logger s_logger;
    return s_logger;
}

} // namespace helix
