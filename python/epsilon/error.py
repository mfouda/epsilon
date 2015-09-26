from epsilon.expression_str import problem_str, expr_str

class ProblemError(Exception):
    def __init__(self, message, problem):
        super(ProblemError, self).__init__(message)
        self.problem = problem

    def __str__(self):
        return (super(ProblemError, self).__str__() + "\n" +
                problem_str(self.problem))

class ExpressionError(Exception):
    def __init__(self, message, expr):
        super(ExpressionError, self).__init__(message)
        self.expr = expr

    def __str__(self):
        return (super(ExpressionError, self).__str__() + "\n" +
                expr_str(self.expr))
