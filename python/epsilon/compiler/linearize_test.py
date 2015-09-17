
from epsilon.expression import *
from epsilon.compiler import linearize

m = 10
n = 5
x = variable(n, 1, "x")

def test_constraint_with_constant():
    A = constant(m, n)
    b = constant(m, 1)
    prob = linearize.transform(
        Problem(objective=add(equality_constraint(multiply(A, x), b))))

    assert len(prob.constraint) == 0
    assert prob.objective.expression_type == Expression.ADD
    assert len(prob.objective.arg) == 1
    assert prob.objective.arg[0].expression_type == Expression.INDICATOR
                        
def test_constraint_no_constant():
    prob = linearize.transform(
        Problem(objective=add(equality_constraint(x, x))))

    assert len(prob.constraint) == 1
    assert prob.constraint[0].expression_type == Expression.INDICATOR
    assert prob.objective.expression_type == Expression.ADD    
    assert len(prob.objective.arg) == 1
    assert prob.objective.arg[0].expression_type == Expression.CONSTANT
    
    


