#ifndef EPSILON_EXPRESSION_VAR_OFFSET_MAP_H
#define EPSILON_EXPRESSION_VAR_OFFSET_MAP_H

#include <unordered_map>

#include "epsilon/expression.pb.h"

class VariableOffsetMap {
 public:
  VariableOffsetMap() : n_(0) {}

  // Insert all variables from problem/expression
  void Insert(const Expression& expr);
  void Insert(const Problem& problem);

  int Get(const std::string& var_id) const;
  int Size(const std::string& var_id) const;
  int n() const { return n_; }
  std::unordered_map<std::string, int> offsets() const { return offsets_; }

 private:
  std::unordered_map<std::string, int> offsets_;
  std::unordered_map<std::string, int> sizes_;
  int n_;
};

#endif  // EPSILON_EXPRESSION_VAR_OFFSET_MAP_H
