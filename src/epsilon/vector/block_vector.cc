
#include <set>

#include <glog/logging.h>

#include "epsilon/vector/block_vector.h"
#include "epsilon/vector/vector_util.h"

BlockVector& BlockVector::operator+=(const BlockVector& rhs) {
  for (auto iter : rhs.data_)
    InsertOrAdd(iter.first, iter.second);
  return *this;
}

BlockVector& BlockVector::operator-=(const BlockVector& rhs) {
  for (auto iter : rhs.data_)
    InsertOrAdd(iter.first, -iter.second);
  return *this;
}

BlockVector& BlockVector::operator*=(double alpha) {
  for (auto& iter : data_) {
    iter.second *= alpha;
  }
  return *this;
}

BlockVector operator+(BlockVector lhs, const BlockVector& rhs) {
  lhs += rhs;
  return lhs;
}

BlockVector operator-(BlockVector lhs, const BlockVector& rhs) {
  lhs -= rhs;
  return lhs;
}

BlockVector operator*(double alpha, BlockVector x) {
  x *= alpha;
  return x;
}

void BlockVector::InsertOrAdd(
    const std::string& key,
    DenseVector value) {
  auto res = data_.insert(std::make_pair(key, value));
  if (!res.second) (res.first)->second += value;
}

BlockVector::DenseVector& BlockVector::operator()(const std::string& key) {
  return data_[key];
}

const BlockVector::DenseVector& BlockVector::operator()(
    const std::string& key) const {
  auto iter = data_.find(key);
  if (iter == data_.end())
    LOG(FATAL) << key << " not in BlockVector";
  return iter->second;
}

BlockVector::DenseVector BlockVector::Get(const std::string& key, int n) const {
  auto iter = data_.find(key);
  if (iter == data_.end())
    return DenseVector::Zero(n);
  return iter->second;
}

BlockVector BlockVector::Select(const std::set<std::string>& keys) const {
  BlockVector retval;
  for (const std::string& key : keys) {
    auto iter = data_.find(key);
    if (iter != data_.end())
      retval.data_.insert(*iter);
  }
  return retval;
}

int BlockVector::n() const {
  int n = 0;
  for (const auto& iter : data_) {
    n += iter.second.rows();
  }
  return n;
}

double BlockVector::norm() const {
  double norm_squared = 0;
  for (const auto& iter : data_) {
    norm_squared += iter.second.squaredNorm();
  }
  return sqrt(norm_squared);
}

std::string BlockVector::DebugString() const {
  std::string retval = "";
  for (auto iter : data_) {
    if (retval != "") retval += " ";
    retval += iter.first + ": " + VectorDebugString(iter.second);
  }
  return retval;
}

std::set<std::string> BlockVector::keys() const {
  std::set<std::string> retval;
  for (const auto& iter : data_) {
    retval.insert(iter.first);
  }
  return retval;
}
