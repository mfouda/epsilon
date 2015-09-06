#ifndef EPSILON_UTIL_VECTOR_FILE_H
#define EPSILON_UTIL_VECTOR_FILE_H

#include <memory>
#include <string>

#include <Eigen/Dense>

#include "epsilon/data.pb.h"

// Write path
void WriteMatrixData(const Eigen::MatrixXd& input, const std::string& location);

// Read path is broken up in to two steps currently
std::unique_ptr<const Data> ReadSplitData(const std::string& location);
Eigen::MatrixXd GetMatrixData(const Data& data);

std::string metadata_file(const std::string& location);
std::string value_file(const std::string& location);
std::string value_transpose_file(const std::string& location);

#endif  // EPSILON_UTIL_VECTOR_FILE_H
