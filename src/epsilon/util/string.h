#ifndef UTIL_STRING_H
#define UTIL_STRING_H

#include <stdarg.h>

#include <string>
#include <vector>

std::string StringPrintf(const std::string fmt_str, ...);

void StringReplace(
    const std::string& from, const std::string& to, std::string* str);

std::vector<std::string> Split(const std::string& input, char delim);
std::string Join(
  const std::vector<std::string>::const_iterator& begin,
  const std::vector<std::string>::const_iterator& end,
  const std::string& delim);

bool StartsWith(const std::string& prefix, const std::string& str);

#endif  // UTIL_STRING_H
