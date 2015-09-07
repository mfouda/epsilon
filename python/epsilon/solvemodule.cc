#include <Python.h>

#include "epsilon/file/file.h"
#include "epsilon/prox.pb.h"
#include "epsilon/solver_params.pb.h"

extern "C" {

// solve(problem_str, params_str, data)
static PyObject* solve_admm_prox(PyObject* self, PyObject* args) {
  const char* problem_str;
  const char* params_str;
  PyObject* data;
  if (!PyArg_ParseTuple(args, "ssO", &problem_str, &params_str, &data))
    return nullptr;

  ProxProblem problem;
  SolverParams params;

  if (!problem.ParseFromString(problem_str))
    return nullptr;
  if (!params.ParseFromString(params_str))
    return nullptr;

  Py_ssize_t pos = 0;
  PyObject* key;
  PyObject* value;
  while (PyDict_Next(data, &pos, &key, &value)) {
    const char* key_str = PyString_AsString(key);
    if (!key_str)
      return nullptr;

    const char* value_str = PyBytes_AsString(value);
    if (!value_str)
      return nullptr;

    {
      std::unique_ptr<file::File> f = file::Open(key_str, file::kWriteMode);
      f->Write(std::string(value_str, PyString_Size(value)));
      f->Close();
    }
  }

  return nullptr;
}

static PyMethodDef SolveMethods[] = {
  {"solve_admm_prox", solve_admm_prox, METH_VARARGS,
   "Solve a problem with epsilon."},
  {nullptr, nullptr, 0, nullptr}
};

PyMODINIT_FUNC init_solve() {
  (void)Py_InitModule("_solve", SolveMethods);
}

}  // extern "C"
