#include "distopt/expression/problem.h"

#include <unordered_map>

#include <glog/logging.h>

#include "distopt/problem.pb.h"
#include "distopt/expression/expression.h"


int VariableOffsetMap::Get(const Expression& expr) {
  CHECK_EQ(expr.expression_type(), Expression::VARIABLE);
  CHECK(expr.variable().variable_id() != "");
  const std::string id = expr.variable().variable_id();
  auto iter = offsets_.find(id);
  int offset;
  if (iter == offsets_.end()) {
    offset = n_;
    offsets_.insert(make_pair(id, n_));
    n_ += GetDimension(expr);
  } else {
    offset = iter->second;
  }
  return offset;
}

void AddVariableOffsets(Problem* problem) {
  VariableOffsetMap var_offsets;
  AddVariableOffsets(problem->mutable_objective(), &var_offsets);
  for (Constraint& constr : *problem->mutable_constraint()) {
    for (Expression& arg : *constr.mutable_arg()) {
      AddVariableOffsets(&arg, &var_offsets);
    }
  }
  problem->set_n(var_offsets.n());
}

void AddVariableOffsets(ProxProblem* problem) {
  VariableOffsetMap var_offsets;
  for (ProxFunction& f : *problem->mutable_prox_function()) {
    for (Expression& arg : *f.mutable_arg())
      AddVariableOffsets(&arg, &var_offsets);
    if (f.has_affine())
      AddVariableOffsets(f.mutable_affine(), &var_offsets);
    if (f.has_regularization())
      AddVariableOffsets(f.mutable_regularization(), &var_offsets);
  }
  for (Expression& arg : *problem->mutable_equality_constraint()) {
    AddVariableOffsets(&arg, &var_offsets);
  }
}

void AddVariableOffsets(ProxFunction* f) {
  VariableOffsetMap var_offsets;
  for (Expression& arg : *f->mutable_arg())
    AddVariableOffsets(&arg, &var_offsets);
  if (f->has_affine())
    AddVariableOffsets(f->mutable_affine(), &var_offsets);
  if (f->has_regularization())
    AddVariableOffsets(f->mutable_regularization(), &var_offsets);
}

int GetConstraintDimension(const Constraint& constr) {
  int dim = 0;
  for (const Expression& arg : constr.arg()) {
    dim += GetDimension(arg);
  }
  return dim;
}

std::vector<const Expression*> GetVariables(const Problem& problem) {
  std::set<const Expression*, VariableIdCompare> retval;
  GetVariables(problem.objective(), &retval);
  for (const Constraint& constr : problem.constraint()) {
    for (const Expression& arg : constr.arg()) {
      GetVariables(arg, &retval);
    }
  }

  return {retval.begin(), retval.end()};
}

std::vector<const Expression*> GetVariables(const ProxProblem& problem) {
  std::set<const Expression*, VariableIdCompare> retval;
  for (const ProxFunction& f : problem.prox_function()) {
    for (const Expression& arg : f.arg())
      GetVariables(arg, &retval);
    GetVariables(f.affine(), &retval);
    GetVariables(f.regularization(), &retval);
  }
  for (const Expression& constr : problem.equality_constraint())
    GetVariables(constr, &retval);

  return {retval.begin(), retval.end()};
}

std::vector<const Expression*> GetVariables(const ProxFunction& f) {
  std::set<const Expression*, VariableIdCompare> retval;
  for (const Expression& arg : f.arg())
    GetVariables(arg, &retval);
  GetVariables(f.affine(), &retval);
  GetVariables(f.regularization(), &retval);
  return {retval.begin(), retval.end()};
}

void GetProblemDimensions(const Problem& problem, int* m, int* n) {
  if (n != nullptr) {
    *n = 0;
    for (const Expression* expr : GetVariables(problem)) {
      *n += GetDimension(*expr);
    }
  }

  if (m != nullptr) {
    *m = 0;
    for (const Constraint& constr : problem.constraint()) {
      for (const Expression& expr : constr.arg()) {
        *m += GetDimension(expr);
      }
    }
  }
}

int GetVariableDimension(const ProxProblem& problem) {
  int n = 0;
  for (const Expression* expr : GetVariables(problem)) {
    n += GetDimension(*expr);
  }
  return n;
}

uint64_t VariableId(uint64_t problem_id, const std::string& variable_id_str) {
  return problem_id ^ std::hash<std::string>()(variable_id_str);
}
