
#ifndef EPSILON_VECTOR_BLOCK_VECTOR_H
#define EPSILON_VECTOR_BLOCK_VECTOR_H

#include <Eigen/Dense>

#include <glog/logging.h>

#include <map>

class BlockMatrix;

class BlockVector {
 public:
  typedef Eigen::Matrix<double, Eigen::Dynamic, 1> DenseVector;

  BlockVector() {
    VLOG(3) << "default ctor";
  }

  ~BlockVector() {
    VLOG(3) << "dtor";
  }

  BlockVector(const BlockVector& rhs) {
    VLOG(3) << "copy ctor";
    data_ = rhs.data_;
  }

  BlockVector(BlockVector&& rhs) {
    VLOG(3) << "move ctor";
    data_ = std::move(rhs.data_);
  }

  BlockVector& operator=(const BlockVector& rhs) {
    VLOG(3) << "copy assignment";
    BlockVector lhs(rhs);
    *this = std::move(lhs);
    return *this;
  }

  BlockVector& operator=(BlockVector&& rhs) {
    VLOG(3) << "move assignment";
    std::swap(data_, rhs.data_);
    return *this;
  }

  DenseVector& operator()(const std::string& key);
  const DenseVector& operator()(const std::string& key) const;

  // Gets the desired key or returns a vector of zeros
  DenseVector Get(const std::string& key, int n) const;

  BlockVector& operator+=(const BlockVector& rhs);
  BlockVector& operator-=(const BlockVector& rhs);
  BlockVector& operator*=(double alpha);

  friend BlockVector operator*(const BlockMatrix& A, const BlockVector& x);

  int n() const;
  double norm() const;
  bool has_key(const std::string& key) const {
    return data_.find(key) != data_.end();
  }
  const std::map<std::string, DenseVector>& data() const { return data_; }


  std::string DebugString() const;
  void InsertOrAdd(const std::string& key, DenseVector value);

 private:
  std::map<std::string, DenseVector> data_;
};

BlockVector operator+(BlockVector lhs, const BlockVector& rhs);
BlockVector operator-(BlockVector lhs, const BlockVector& rhs);
BlockVector operator*(double alpha, BlockVector rhs);

#endif  // EPSILON_VECTOR_BLOCK_VECTOR_H
