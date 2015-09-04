#include "epsilon/util/string.h"

#include <string.h>

#include <cstdlib>
#include <memory>
#include <regex>
#include <sstream>
#include <string>

using std::string;
using std::unique_ptr;

string StringPrintf(const string fmt_str, ...) {
  int final_n, n = ((int)fmt_str.size()) * 2;
  string str;
  unique_ptr<char[]> formatted;
  va_list ap;

  while(1) {
    formatted.reset(new char[n]);
    strcpy(&formatted[0], fmt_str.c_str());
    va_start(ap, fmt_str);
    final_n = vsnprintf(&formatted[0], n, fmt_str.c_str(), ap);
    va_end(ap);
    if (final_n < 0 || final_n >= n)
      n += abs(final_n - n + 1);
    else
      break;
  }

  return string(formatted.get());
}

void StringReplace(
    const std::string& from, const std::string& to, std::string* str) {
  size_t start_pos = str->find(from);
  if (start_pos != std::string::npos) {
    str->replace(start_pos, from.length(), to);
  }
}

std::vector<std::string> Split(const std::string &s, char delim) {
  std::vector<std::string> elems;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, delim)) {
    elems.push_back(item);
  }
  return elems;
}

std::string Join(
  const std::vector<std::string>::const_iterator& begin,
  const std::vector<std::string>::const_iterator& end,
  const std::string& delim) {
  string retval;
  for (auto iter = begin; iter != end; ++iter) {
    if (!retval.empty()) retval += delim;
    retval += *iter;
  }
  return retval;
}

bool StartsWith(const std::string& prefix, const std::string& str) {
  return std::mismatch(prefix.begin(), prefix.end(), str.begin()).first ==
      prefix.end();
}
