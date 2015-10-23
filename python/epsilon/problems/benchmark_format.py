

from collections import namedtuple

class Column(namedtuple("Column", ["name", "width", "fmt", "right", "colspan"])):
    """Columns for a Markdown appropriate text table."""

    @property
    def header(self):
        align = "" if self.right else "-"
        header_fmt = " %" + align + str(self.width-2) + "s "
        return header_fmt % self.name

    @property
    def sub_header(self):
        val = "-" * (self.width-2)
        if self.right:
            val = " " + val + ":"
        else:
            val = ":" + val + " "
        return val

    def format(self, data):
        return self.fmt % data

Column.__new__.__defaults__ = (None, None, None, False, 1)

class Formatter(object):
    def __init__(self, super_columns, columns):
        self.super_columns = super_columns
        self.columns = columns

    def print_header(self):
        pass

    def print_footer(self):
        pass

class Text(Formatter):
    def print_header(self):
        print "|".join(c.header for c in self.super_columns)
        print "|".join(c.header for c in self.columns)
        print "|".join(c.sub_header for c in self.columns)

    def print_row(self, data):
        print "|".join(c.fmt % data[i] for i, c in enumerate(self.columns))

class HTML(Formatter):
    def print_header(self):
        print "<table>"
        print "<tr>" + "".join('<th colspan="%d">%s</th>' % (c.colspan, c.header)
                               for c in self.super_columns) + "</tr>"
        print "<tr>" + "".join('<th colspan="%d">%s</th>' % (c.colspan, c.header)
                               for c in self.columns) + "</tr>"

    def print_row(self, data):
        print "".join("<td>" + c.fmt % data[i] + "</td>"
                      for i, c in enumerate(self.columns))

    def print_footer(self):
        print "</table>"

def format_sci_latex(s):
    if "e+" in s or "e-" in s:
        k, exp = s.split("e")
        if exp[1].strip() == '0':
            exp = exp[0] + exp[2]
        if exp[0] == '+':
            exp = exp[1:]
        return r"$%s \times 10^{%s}$" % (k, exp)
    else:
        return s

class Latex(Formatter):
    def print_header(self):
        print r"\begin{tabular}"
        print "&".join("\multicolumn{%d}{c}{%s}" % (c.colspan, c.header)
                       if c.colspan != 1
                       else c.header
                       for c in self.super_columns) + r" \\"

        print "&".join("\multicolumn{%d}{c}{%s}" % (c.colspan, c.header)
                       if c.colspan != 1
                       else c.header
                       for c in self.columns) + r" \\"


    def print_row(self, data):
        print ("&".join(
            [r"\texttt{%s}" % data[0].replace("_", "\_")] +
            [format_sci_latex(c.fmt % data[i+1])
             for i, c in enumerate(self.columns[1:])])
        + r" \\")

    def print_footer(self):
        print r"\end{tabular}"
