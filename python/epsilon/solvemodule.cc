#include <Python.h>

extern "C" {
static PyObject* solve(PyObject* self, PyObject* args);

static PyMethodDef SolveMethods[] = {
  {"solve", solve, METH_VARARGS,
   "Solve a problem with epsilon."},
  {nullptr, nullptr, 0, nullptr}
};

static PyObject* solve(PyObject* self, PyObject* args) {
  return nullptr;
}

PyMODINIT_FUNC initsolve() {
  (void)Py_InitModule("solve", SolveMethods);
}

}
