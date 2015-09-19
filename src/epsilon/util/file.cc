#include "epsilon/util/file.h"

#include <fstream>

#include <glog/logging.h>

std::string ReadStringFromFile(const std::string& filename) {
  std::ifstream in(filename, std::ios::in | std::ios::binary);
  CHECK(in);

  std::string contents;
  in.seekg(0, std::ios::end);
  contents.resize(in.tellg());
  in.seekg(0, std::ios::beg);
  in.read(&contents[0], contents.size());
  in.close();
  return contents;
}
