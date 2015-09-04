#ifndef EPSILON_FILE_FILE_H
#define EPSILON_FILE_FILE_H

#include <memory>
#include <string>
#include <vector>

namespace file {

const std::string kReadMode = "r";
const std::string kWriteMode = "w";

void Init();
void Cleanup();

class File {
public:
  virtual ~File() {}

  virtual void Open() = 0;
  virtual std::string Read(
      size_t position = 0, size_t length = std::string::npos) = 0;
  virtual void Write(const std::string& contents) = 0;
  virtual void Close() = 0;
};

std::unique_ptr<File> Open(const std::string& name, const std::string& mode = kReadMode);

}  // namespace file

#endif  // EPSILON_FILE_FILE_H
