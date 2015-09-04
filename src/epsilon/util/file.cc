
#include "distopt/util/file.h"

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

void WriteStringToFile(
  const std::string& contents, const std::string& filename) {
  std::ofstream out(filename, std::ios::out | std::ios::binary);
  CHECK(out);

  out.write(&contents[0], contents.size());
  out.close();
}
