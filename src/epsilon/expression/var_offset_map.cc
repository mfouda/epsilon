#include "epsilon/expression/var_offset_map.h"

#include <glog/logging.h>

#include "epsilon/expression/expression_util.h"

void VariableOffsetMap::Insert(const Expression& expr) {
  if (expr.expression_type() == Expression::VARIABLE) {
    const std::string& var_id = expr.variable().variable_id();
    auto iter = offsets_.find(var_id);
    if (iter == offsets_.end()) {
      int size = GetDimension(expr);
      offsets_.insert(make_pair(var_id, n_));
      sizes_.insert(make_pair(var_id, size));
      n_ += size;
    }
  }

  for (const Expression& arg : expr.arg())
    Insert(arg);
}

void VariableOffsetMap::Insert(const Problem& problem) {
  Insert(problem.objective());
  for (const Expression& constr : problem.constraint())
    Insert(constr);
}

int VariableOffsetMap::Get(const std::string& var_id) const {
  auto iter = offsets_.find(var_id);
  CHECK(iter != offsets_.end());
  return iter->second;
}

bool VariableOffsetMap::Contains(const std::string& var_id) const {
  auto iter = offsets_.find(var_id);
  return iter != offsets_.end();
}

int VariableOffsetMap::Size(const std::string& var_id) const {
  auto iter = sizes_.find(var_id);
  CHECK(iter != sizes_.end());
  return iter->second;
}
