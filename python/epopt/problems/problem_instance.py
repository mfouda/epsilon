
from collections import namedtuple

class ProblemInstance(namedtuple("ProblemInstance", ["name", "f", "kwargs"])):
    def create(self):
        return self.f(**self.kwargs)
