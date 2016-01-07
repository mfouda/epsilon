#ifndef UTIL_LOGGING_H
#define UTIL_LOGGING_H

#include <string>

typedef void(*LogFunction)(const std::string& msg);

void LogVerbose(const std::string& msg);
void SetVerboseLogger(LogFunction f);

#endif  // UTIL_LOGGING_H
