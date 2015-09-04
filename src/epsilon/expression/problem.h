#ifndef EXPRESSION_PROBLEM_H
#define EXPRESSION_PROBLEM_H

#include <glog/logging.h>

#include "epsilon/prox.pb.h"


// Adds variable offets to the problem variables
void AddVariableOffsets(ProxProblem* problem);
void AddVariableOffsets(ProxFunction* function);

// Gets the row dimension of A in the Ax + s = b constraint.
int GetVariableDimension(const ProxProblem& problem);

std::vector<const Expression*> GetVariables(const ProxProblem& problem);
std::vector<const Expression*> GetVariables(const ProxFunction& function);

uint64_t VariableId(uint64_t problem_id, const std::string& variable_id_str);

#endif  // EXPRESSION_PROBLEM_H
