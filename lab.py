"""
6.101 Lab:
LISP Interpreter Part 2
"""

#!/usr/bin/env python3
import sys

sys.setrecursionlimit(20_000)


# NO ADDITIONAL IMPORTS!

#############################
# Scheme-related Exceptions #
#############################


class SchemeError(Exception):
    """
    A type of exception to be raised if there is an error with a Scheme
    program.  Should never be raised directly; rather, subclasses should be
    raised.
    """

    pass


class SchemeSyntaxError(SchemeError):
    """
    Exception to be raised when trying to evaluate a malformed expression.
    """

    pass


class SchemeNameError(SchemeError):
    """
    Exception to be raised when looking up a name that has not been defined.
    """

    pass


class SchemeEvaluationError(SchemeError):
    """
    Exception to be raised if there is an error during evaluation other than a
    SchemeNameError.
    """

    pass


############################
# Tokenization and Parsing #
############################


def number_or_symbol(value):
    """
    Helper function: given a string, convert it to an integer or a float if
    possible; otherwise, return the string itself

    >>> number_or_symbol('8')
    8
    >>> number_or_symbol('-5.32')
    -5.32
    >>> number_or_symbol('1.2.3.4')
    '1.2.3.4'
    >>> number_or_symbol('x')
    'x'
    """
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def tokenize(source):
    """
    Splits an input string into meaningful tokens (left parens, right parens,
    other whitespace-separated values).  Returns a list of strings.

    Arguments:
        source (str): a string containing the source code of a Scheme
                      expression
    """
    tokens = []

    def sep_parentheses(string):
        """returns a list of tokens"""
        if not string:
            return []

        if "(" not in string and ")" not in string:
            return [string]
        if string[0] in "()":
            return [string[0]] + sep_parentheses(string[1:])
        else:
            tokens = []
            str_token = ""
            for i, char in enumerate(string):
                if char not in "()":
                    str_token += char
                else:
                    tokens.append(str_token)
                    break
            tokens += sep_parentheses(string[i:])
            return tokens

    for line in source.split("\n"):
        if not line or ";" == line[0]:
            # a comment is a line starting with ";"
            continue
        if ";" in line:
            line = line[: line.index(";")]
        for _, token in enumerate(line.split(" ")):
            if "(" in token or ")" in token:
                tokens += sep_parentheses(token)

            elif token:
                tokens.append(token)

    return tokens


def parse(tokens):
    """
    Parses a list of tokens, constructing a representation where:
        * symbols are represented as Python strings
        * numbers are represented as Python ints or floats
        * S-expressions are represented as Python lists

    Arguments:
        tokens (list): a list of strings representing tokens
    """

    def parse_expression(index):
        if tokens[index] == ")":
            raise SchemeSyntaxError("Expression not well formed!")
        if tokens[index] == "(":
            exp = []
            next_index = index + 1
            try:
                while tokens[next_index] != ")":
                    result, next_index = parse_expression(next_index)
                    exp.append(result)
                return exp, next_index + 1
            except IndexError as exc:
                raise SchemeSyntaxError("Expression not well formed!") from exc
        else:
            return number_or_symbol(tokens[index]), index + 1

    output_tokens, last_index = parse_expression(0)
    if last_index < len(tokens):
        raise SchemeSyntaxError("Expression not well formed!")
    return output_tokens


######################
# Built-in Functions #
######################


def multiply(nums):
    result = 1
    for num in nums:
        result *= num
    return result


def divide(nums):
    result = nums[0]
    for num in nums[1:]:
        result = result / num
    return result


def negate(args):
    if len(args) != 1:
        raise SchemeEvaluationError(f"'not' takes in 1 argument but {len(args)} given")
    return not bool(args[0])


def linked_list(args):
    return Pair(args[0], linked_list(args[1:])) if args else []


def length(ll):
    if ll == []:
        return 0
    if isinstance(ll, Pair):
        return ll.length()
    raise SchemeEvaluationError(f"{ll} Not a list!")


def is_list(obj):
    while obj != []:
        if not isinstance(obj, Pair):
            return False
        obj = obj.cdr
    return True


def list_ref(ll, index):
    if not isinstance(ll, Pair) or not isinstance(index, int):
        raise SchemeEvaluationError("Not a list, or index out of range!")
    if index == 0:
        return ll.car
    return list_ref(ll.cdr, index-1)


def copy_list(ll, elt=[]):
    if ll == []:
        return elt
    if isinstance(ll, Pair):
        return Pair(ll.car, copy_list(ll.cdr, elt))
    raise SchemeEvaluationError("Not a list!")


def concatenate(lists=[]):
    if not lists:
        return []
    return copy_list(lists[0], concatenate(lists[1:]))


def car(args):
    if len(args) != 1:
        raise SchemeEvaluationError
    if isinstance(args[0], Pair):
        return args[0].car
    raise SchemeEvaluationError


def cdr(args):
    if len(args) != 1:
        raise SchemeEvaluationError
    if isinstance(args[0], Pair):
        return args[0].cdr
    raise SchemeEvaluationError


def begin(args):
    if len(args) > 0:
        return args[-1]
    raise SchemeEvaluationError


##############
# Frame      #
##############


class Pair:
    """Pair object class used to construct lists"""
    def __init__(self, car, cdr) -> None:
        self.car = car
        self.cdr = cdr

    def __str__(self):
        return f"Pair({self.car}, {self.cdr})"

    def length(self):
        if isinstance(self.cdr, Pair):
            return self.cdr.length() + 1
        if self.cdr == []:
            return 1
        raise SchemeEvaluationError("Not a list!")

    def get(self, index):
        if index > 0 and isinstance(self.cdr, Pair):
            return self.cdr.get(index - 1)
        if index == 0:
            return self.car
        raise SchemeEvaluationError


class Builtins:
    """
    Builtins class consists of scheme builtin functions
    """

    def __init__(self):
        self.variables = {
            "+": sum,
            "-": lambda args: -args[0] if len(args) == 1 else (args[0] - sum(args[1:])),
            "*": multiply,
            "/": divide,
            "equal?": lambda args: all(val == args[0] for val in args),
            ">": lambda args: all(args[i] > args[i + 1] for i in range(len(args) - 1)),
            ">=": lambda args: all(
                args[i] >= args[i + 1] for i in range(len(args) - 1)
            ),
            "<": lambda args: all(args[i] < args[i + 1] for i in range(len(args) - 1)),
            "<=": lambda args: all(
                args[i] <= args[i + 1] for i in range(len(args) - 1)
            ),
            "not": negate,
            "#t": True,
            "#f": False,
            "car": car,
            "cdr": cdr,
            "list": linked_list,
            "list?": lambda args: is_list(args[0]),
            "length": lambda args: length(args[0]),
            "list-ref": list_ref,
            "append": concatenate,
            "begin": begin,
        }

    def get(self, variable):
        if variable in self.variables:
            return self.variables[variable]
        else:
            raise SchemeNameError(f"Variable not found! <{variable=}>")


class Frame:
    """
    Frame consists of local variables
    and variables inherited from the parent class
    """

    def __init__(self, parent=Builtins()):
        self.__parent = parent
        self.__variables = {}

    def __str__(self):
        if isinstance(self.__parent, Builtins):
            return f"Frame: {self.__variables}"
        return f"Frame: {self.__variables} {self.__parent}"

    def set(self, variable, value):
        self.__variables[variable] = value

    def get(self, variable):
        if variable in self.__variables:
            return self.__variables[variable]
        return self.__parent.get(variable)

    def delete(self, variable):
        if variable in self.__variables:
            value = self.__variables[variable]
            del self.__variables[variable]
            return value
        raise SchemeNameError(f"Variable '{variable}' not in current frame!")

    def setbang(self, variable, value):
        if variable in self.__variables:
            self.__variables[variable] = value
        elif isinstance(self.__parent, Frame):
            self.__parent.setbang(variable, value)
        else:
            raise SchemeNameError


def make_initial_frame():
    return Frame(parent=Builtins())


class CustomFunction:
    """
    Custom functions class:
    Stores parameters, expression, and pointer to enclosing frame
    """

    def __init__(self, parameters, expression, enclosing_frame):
        self.parameters = parameters
        self.expression = expression
        self.frame = enclosing_frame

    def __str__(self):
        return f"CustomFunction: {self.parameters} {self.expression}"

    def call(self, arguments):
        """
        Calls the function and returns the result of the evaluated expession
        """
        if len(arguments) != len(self.parameters):
            raise SchemeEvaluationError("Wrong number of arguments given!")

        new_args = [evaluate(arg, self.frame) for arg in arguments]

        new_frame = Frame(self.frame)
        for param, arg in zip(self.parameters, new_args):
            new_frame.set(param, arg)

        return evaluate(self.expression, new_frame)

    def __call__(self, arguments):
        return self.call(arguments)


##############
# Lists #
##############


def cons(car, cdr):
    return Pair(car, cdr)


def list_operations(tree, frame):
    args = [evaluate(item, frame) for item in tree[1:]]
    match tree[0]:
        case "cons" if len(tree) == 3:
            first = evaluate(tree[1], frame)
            rest = evaluate(tree[2], frame)
            return Pair(first, rest)
        case "list" | "append":
            func = frame.get(tree[0])
            return func(args)
        case "list-ref" if len(tree) == 3:
            return list_ref(*args)
        case "list?" | "length" | "car" | "cdr" if len(tree) == 2:
            func = frame.get(tree[0])
            return func(args)

    raise SchemeEvaluationError(f"{tree[0]=} ")


##############
# Evaluation #
##############


def define(tree, frame=None):
    if isinstance(tree[1], str):
        value = evaluate_recursive(tree[2], frame)
        frame.set(tree[1], value)
        return value
    if isinstance(tree[1], list):  # Simplified function definition
        func = CustomFunction(tree[1][1:], tree[2], frame)
        frame.set(tree[1][0], func)
        return func


def apply_special_form(tree, frame=None):
    match tree[0]:
        case "define" if len(tree) == 3:
            return define(tree, frame)
        case "lambda":
            return CustomFunction(tree[1], tree[2], frame)
        case "and":
            for arg in tree[1:]:
                if not evaluate_recursive(arg, frame):
                    return False
            return True
        case "or":
            for arg in tree[1:]:
                if evaluate_recursive(arg, frame):
                    return True
            return False
        case "if":
            pred = evaluate_recursive(tree[1], frame)
            if pred:
                return evaluate_recursive(tree[2], frame)
            return evaluate_recursive(tree[3], frame)
    raise SchemeEvaluationError(f"{tree[0]} is not a function!")


def del_let_set(tree, frame=None):
    match tree[0]:
        case "del" if len(tree) == 2 and isinstance(tree[1], str):
            return frame.delete(tree[1])

        case "let" if len(tree) == 3:
            new_frame = Frame(frame)
            for var, val in tree[1]:
                new_frame.set(var, evaluate_recursive(val, frame))
            return evaluate_recursive(tree[2], new_frame)

        case "set!" if len(tree) == 3 and isinstance(tree[1], str):
            value = evaluate_recursive(tree[2], frame)
            frame.setbang(tree[1], value)
            return value
    raise SchemeEvaluationError(f"{tree[0]} is not a function!")


def evaluate_recursive(tree, frame=None):
    if isinstance(tree, str):
        return frame.get(tree)  # Variable lookup
    if not tree or not isinstance(tree, list):
        return tree

    command = tree[0]

    if isinstance(command, str):
        match command:
            case "define" | "lambda" | "and" | "or" | "if":
                return apply_special_form(tree, frame)
            case "cons" | "car" | "cdr" | "list?" | "length" | "list-ref" | "append":
                return list_operations(tree, frame)
            case "del" | "let" | "set!":
                return del_let_set(tree, frame)

    try:  # Function call
        op = evaluate_recursive(tree[0], frame)
        args = [evaluate_recursive(val, frame) for val in tree[1:]]
        if callable(op):
            return op(args)
    except TypeError:
        return op(*args)

    raise SchemeEvaluationError(f"{tree[0]} is Not a function!")


def evaluate(tree, frame=None):
    """
    Evaluate the given syntax tree according to the rules of the Scheme
    language.

    Arguments:
        tree (type varies): a fully parsed expression, as the output from the
                            parse function
    """
    if not frame:
        frame = make_initial_frame()
    return evaluate_recursive(tree, frame)


def evaluate_file(file_name, frame=None):
    file = open(file_name, "r")
    source = file.read(-1)
    return evaluate(parse(tokenize(source)), frame)


if __name__ == "__main__":
    # NOTE THERE HAVE BEEN CHANGES TO THE REPL, KEEP THIS CODE BLOCK AS WELL
    # code in this block will only be executed if lab.py is the main file being
    # run (not when this module is imported)

    imported_files = sys.argv[1:]
    global_frame = make_initial_frame()
    for file in imported_files:
        evaluate_file(file, global_frame)

    evaluate_file("test_files/map_filter_reduce.scm", global_frame)

    import os

    sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
    import schemerepl

    schemerepl.SchemeREPL(
        sys.modules[__name__], use_frames=True, verbose=True, global_frame=global_frame
    ).cmdloop()
