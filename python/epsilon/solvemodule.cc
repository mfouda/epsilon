#include <Python.h>

#include <setjmp.h>
#include <stdlib.h>

#include <glog/logging.h>

#include "epsilon/algorithms/prox_admm.h"
#include "epsilon/expression.pb.h"
#include "epsilon/expression/expression_util.h"
#include "epsilon/file/file.h"
#include "epsilon/parameters/local_parameter_service.h"
#include "epsilon/solver_params.pb.h"
#include "epsilon/prox/prox.h"

// TODO(mwytock): Does failure handling need to be made threadsafe? Seems like
// making these threadlocal would do
static jmp_buf failure_buf;
static PyObject* SolveError;

BlockVector GetVariableVector(PyObject* vars) {
  BlockVector x;

  Py_ssize_t pos = 0;
  PyObject* key;
  PyObject* value;
  while (PyDict_Next(vars, &pos, &key, &value)) {
    const char* key_str = PyString_AsString(key);
    char* value_str;
    Py_ssize_t value_str_len;
    PyString_AsStringAndSize(value, &value_str, &value_str_len);
    CHECK(key_str != nullptr);
    CHECK(value_str != nullptr);
    CHECK(value_str_len % sizeof(double) == 0);

    x(key_str) = Eigen::Map<const Eigen::VectorXd>(
        reinterpret_cast<const double*>(value_str),
        value_str_len/sizeof(double));
  }
  return x;
}

PyObject* GetVariableMap(const BlockVector& x) {
  PyObject* vars = PyDict_New();
  for (auto iter : x.data()) {
    PyObject* val = PyString_FromStringAndSize(
        reinterpret_cast<const char*>(iter.second.data()),
        iter.second.rows()*sizeof(double));
    PyDict_SetItemString(vars, iter.first.c_str(), val);
    Py_DECREF(val);
  }
  return vars;
}

extern "C" {

void WriteConstants(PyObject* data) {
  // NOTE(mwytock): References returned by PyDict_Next() are borrowed so no need
  // to Py_DECREF() them.
  Py_ssize_t pos = 0;
  PyObject* key;
  PyObject* value;
  while (PyDict_Next(data, &pos, &key, &value)) {
    const char* key_str = PyString_AsString(key);
    const char* value_str = PyString_AsString(value);
    CHECK(key_str);
    CHECK(value_str);

    {
      std::unique_ptr<file::File> f = file::Open(key_str, file::kWriteMode);
      f->Write(std::string(value_str, PyString_Size(value)));
      f->Close();
    }
  }
}


static PyObject* ProxADMMSolve(PyObject* self, PyObject* args) {
  const char* problem_str;
  const char* params_str;
  int problem_str_len, params_str_len;
  PyObject* data;

  // prox_admm_solve(problem_str, params_str, data)
  if (!PyArg_ParseTuple(
          args, "s#s#O",
          &problem_str, &problem_str_len,
          &params_str, &params_str_len,
          &data)) {
    // TODO(mwytock): Need to set the appropriate exceptions when passed
    // incorrect arguments.
    return nullptr;
  }

  Problem problem;
  SolverParams params;
  if (!problem.ParseFromArray(problem_str, problem_str_len))
    return nullptr;
  if (!params.ParseFromArray(params_str, params_str_len))
    return nullptr;

  WriteConstants(data);
  ProxADMMSolver solver(
      problem, params,
      std::unique_ptr<ParameterService>(new LocalParameterService));

  if (!setjmp(failure_buf)) {
    // Standard execution path
    solver.Solve();
    std::string status_str = solver.status().SerializeAsString();

    // Get results
    PyObject* vars = PyDict_New();
    {
      LocalParameterService parameter_service;
      for (const Expression* expr : GetVariables(problem)) {
        const std::string& var_id = expr->variable().variable_id();
        uint64_t param_id = VariableParameterId(solver.problem_id(), var_id);
        Eigen::VectorXd x = parameter_service.Fetch(param_id);

        PyObject* val = PyString_FromStringAndSize(
            reinterpret_cast<const char*>(x.data()),
            x.rows()*sizeof(double));

        PyDict_SetItemString(vars, var_id.c_str(), val);
        Py_DECREF(val);
      }
    }

    PyObject* retval = Py_BuildValue("s#O", status_str.data(), status_str.size(), vars);
    Py_DECREF(vars);
    return retval;
  }

  PyErr_SetString(SolveError, "CHECK failed");
  return nullptr;
}

static PyObject* Prox(PyObject* self, PyObject* args) {
  const char* f_expr_str;
  int f_expr_str_len;
  double lambda;
  PyObject* data;
  PyObject* v;

  // prox(expr_str, lambda, data, v_map)
  if (!PyArg_ParseTuple(
          args, "s#dOO",
          &f_expr_str, &f_expr_str_len, &lambda, &data, &v)) {
    // TODO(mwytock): Need to set the appropriate exceptions when passed
    // incorrect arguments.
    return nullptr;
  }

  Expression f_expr;
  if (!f_expr.ParseFromArray(f_expr_str, f_expr_str_len))
    return nullptr;

  // TODO(mwytock): Make this a parameter?
  BlockMatrix A;

  WriteConstants(data);
  std::unique_ptr<BlockVectorOperator> op = CreateProxOperator(
      lambda, A, f_expr);

  if (!setjmp(failure_buf)) {
    op->Init();
    BlockVector x = op->Apply(GetVariableVector(v));
    PyObject* vars = GetVariableMap(x);
    PyObject* retval = Py_BuildValue("O", vars);
    Py_DECREF(vars);
    return retval;
  }

  PyErr_SetString(SolveError, "CHECK failed");
  return nullptr;
}


void HandleFailure() {
  // TODO(mwytock): Dump stack trace here
  longjmp(failure_buf, 1);
}

static PyMethodDef SolveMethods[] = {
  {"prox_admm_solve", ProxADMMSolve, METH_VARARGS,
   "Solve a problem with epsilon."},
  {"prox", Prox, METH_VARARGS,
   "Test a proximal operator."},
  {nullptr, nullptr, 0, nullptr}
};

static bool initialized = false;
PyMODINIT_FUNC init_solve() {
  // TODO(mwytock): Increase logging verbosity based on environment variable
  if (!initialized) {
    const char* v = getenv("EPSILON_VLOG");
    if (v != nullptr)
      FLAGS_v = atoi(v);
    google::InitGoogleLogging("_solve");
    google::LogToStderr();
    //google::InstallFailureFunction(&HandleFailure);

    initialized = true;
  }

  PyObject* m = Py_InitModule("_solve", SolveMethods);
  if (m == nullptr)
    return;

  SolveError = PyErr_NewException(
      const_cast<char*>("_solve.error"), nullptr, nullptr);
  Py_INCREF(SolveError);
  PyModule_AddObject(m, "error", SolveError);
}

}  // extern "C"
