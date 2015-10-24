

from epsilon.expression_pb2 import Expression, Cone
from epsilon import expression_util

NAMES = {
    Expression.VARIABLE: "xyzwvutsrq",
    Expression.CONSTANT: "abcdkeflmn",
}

OPERATOR_NAMES = {
    Expression.ADD: "+",
    Expression.MULTIPLY: "*",
    Expression.NEGATE: "-",
}

class NameMap(object):
    def __init__(self):
        self.name_map = {}
        self.count = {
            Expression.VARIABLE: 0,
            Expression.CONSTANT: 0,
        }

    def name(self, proto):
        fp = expression_util.fp_expr(proto)
        if fp in self.name_map:
            return self.name_map[fp]

        t = proto.expression_type
        name = NAMES[t][self.count[t] % len(NAMES[t])]
        if proto.size.dim[1] != 1:
            name = name.upper()

        self.name_map[fp] = name
        self.count[t] += 1
        return name

def function_name(proto):
    if proto.expression_type in OPERATOR_NAMES:
        return OPERATOR_NAMES[proto.expression_type]
    elif proto.expression_type == Expression.INDICATOR:
        return Cone.Type.Name(proto.cone.cone_type).lower()
    return Expression.Type.Name(proto.expression_type).lower()

def format_params(proto):
    retval = []
    if proto.expression_type == Expression.INDEX:
        for key in proto.key:
            retval += ["%d:%d" % (key.start, key.stop)]
    elif proto.expression_type in (Expression.POWER, Expression.NORM_P):
        retval += [str(proto.p)]
    elif proto.expression_type == Expression.SUM_LARGEST:
        retval += [str(proto.k)]
    elif proto.expression_type == Expression.SCALED_ZONE:
        retval += ["alpha=%.2f" % proto.scaled_zone_params.alpha,
                   "beta=%.2f" % proto.scaled_zone_params.beta,
                   "C=%.2f" % proto.scaled_zone_params.c,
                   "M=%.2f" % proto.scaled_zone_params.m]

    if retval:
        return "[" + ", ".join(retval) + "]"
    else:
        return ""

def format_expr(proto, name_map):
    if proto.expression_type == Expression.CONSTANT:
        if not proto.constant.data_location:
            return "%.2f" % proto.constant.scalar
        return name_map.name(proto)
    if proto.expression_type == Expression.VARIABLE:
        return name_map.name(proto)

    return (function_name(proto) + format_params(proto) +
            "(" + ", ".join(format_expr(arg, name_map)
                            for arg in proto.arg) + ")")

def format_problem(proto):
    name_map = NameMap()

    assert proto.objective.expression_type == Expression.ADD

    output = "problem(+(\n"
    for arg in proto.objective.arg:
        output += "  " + format_expr(arg, name_map) + ",\n"

    if proto.constraint:
        output += "), [\n"
        for constr in proto.constraint:
            output += "  " + format_expr(constr, name_map) + ",\n"
        output += "])"
    else:
        output += "))"

    return output
