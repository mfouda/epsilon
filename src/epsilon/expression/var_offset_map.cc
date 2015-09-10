#include "epsilon/expression/var_offset_map.h"

#include <glog/logging.h>

#include "epsilon/expression/expression_util.h"

void VariableOffsetMap::Insert(const Expression& expr) {
  if (expr.expression_type() == Expression::VARIABLE) {
    const std::string& var_id = expr.variable().variable_id();
    auto iter = offsets_.find(var_id);
    int offset;
    if (iter == offsets_.end()) {
      offset = n_;
      offsets_.insert(make_pair(var_id, n_));
      n_ += GetDimension(expr);
    } else {
      offset = iter->second;
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

int VariableOffsetMap::Get(const Expression& expr) const {
  CHECK_EQ(expr.expression_type(), Expression::VARIABLE);
  const std::string& var_id = expr.variable().variable_id();
  auto iter = offsets_.find(var_id);
  CHECK(iter != offsets_.end());
  return iter->second;
}
