from epsilon.expression_str import expr_str

class CompilerError(Exception):
    def __init__(self, message, expr):
        super(CompilerError, self).__init__(message)
        self.expr = expr

    def __str__(self):
        return super(CompilerError, self).__str__() + "\n" + expr_str(self.expr)
