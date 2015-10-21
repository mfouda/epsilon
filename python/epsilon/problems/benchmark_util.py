
def modify_data_location(expr, f):
    if (expr.expression_type == Expression.CONSTANT and
        expr.constant.data_location != ""):
        expr.constant.data_location = f(expr.constant.data_location)

    for arg in expr.arg:
        modify_data_location(arg, f)

def makedirs_existok(path):
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise

def write_problems(problems, location):
    """Utility function to write problems for analysis."""

    mem_prefix = "/mem/"
    file_prefix = "/local" + location + "/"
    def rewrite_location(name):
        assert name[:len(mem_prefix)] == mem_prefix
        return file_prefix + name[len(mem_prefix):]

    makedirs_existok(location)
    for problem in problems:
        prob_proto, data_map = cvxpy_expr.convert_problem(problem.create())
        prob_proto = compiler.compile(prob_proto)

        modify_data_location(prob_proto.objective, rewrite_location)
        for constraint in prob_proto.constraint:
            modify_data_location(constraint, rewrite_location)

        with open(os.path.join(location, problem.name), "w") as f:
            f.write(prob_proto.SerializeToString())

        for name, value in data_map.items():
            assert name[:len(mem_prefix)] == mem_prefix
            filename = os.path.join(location, name[len(mem_prefix):])
            makedirs_existok(os.path.dirname(filename))
            with open(filename, "w") as f:
                f.write(value)
