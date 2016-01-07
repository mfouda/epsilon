
#include "epsilon/util/logging.h"

static LogFunction verbose_logger = nullptr;

void SetVerboseLogger(LogFunction f) {
  verbose_logger = f;
}

void LogVerbose(const std::string& msg) {
  if (verbose_logger != nullptr)
    verbose_logger(msg);
}
