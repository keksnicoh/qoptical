# -*- coding: utf-8 -*-
""" module to dissassemble functions to generate
    OpenCL functions from it. Currently this module
    is capable of handling a small subset of functions:

        1. Single Expression
        2. No Conditionals
        3. 2 Arguments

    XXX
        - quiet experimental and messy module at the moment.

    :author: keksnicoh
"""

import dis
import math
import numpy as np

T_VAR = 'VAR'
T_FUNC = 'T_FUNC'
T_SYMBOLE = 'T_SYMBOLE'
T_METHOD = 'T_METHOD'
T_GLOBAL_SYMBOLE = 'T_GLOBAL_SYMBOLE'
T_VAL = 'T_VAL'
T_BIN = 'T_BIN'
T_RETURN = 'T_RETURN'
T_DICT = 'T_DICT'
T_DEREF = 'T_DEREF'
T_UNARY_NEGATIVE = 'T_UNARY_NEGATIVE'

GLOBALS_MAP = [
    (np.sin, 'sin'),
    (np.cos, 'cos'),
    (np.tan, 'tan'),
    (np.arcsin, 'asin'),
    (np.arccos, 'acos'),
    (np.arctan, 'atan'),
    (np.sinh, 'sinh'),
    (np.cosh, 'cosh'),
    (np.tanh, 'tanh'),
    (np.sqrt, 'sqrt'),
    (np.exp, 'exp'),
    (np.abs, 'fabs'),
    (np.cbrt, 'cbrt'),
    (math.sin, 'sin'),
    (math.cos, 'cos'),
    (math.tan, 'tan'),
    (math.asin, 'asin'),
    (math.acos, 'acos'),
    (math.atan, 'atan'),
    (math.sinh, 'sinh'),
    (math.cosh, 'cosh'),
    (math.tanh, 'tanh'),
    (math.sqrt, 'sqrt'),
    (math.exp, 'exp'),
]


def r_clfloat(f, prec=None):
    """ renders an OpenCL float representation
        """
    if prec is not None:
        raise NotImplementedError()
    sf = '{:f}'.format(f)
    return ''.join([sf, '' if '.' in sf else '.', 'f'])


def r_clint(f):
    """ renders an OpenCL integer representation
        """
    assert isinstance(f, int)
    return str(f)


def r_clfrac(p, q, prec=None):
    """ renders an OpenCL fractional representation
        """
    return "{}/{}".format(r_clfloat(p, prec), r_clfloat(q, prec))


def f2cl(f, cl_fname, cl_param_type=None):
    """ converts given function **f** to OpenCL function
        with name **cl_fname**.
        """
    ctree = create_ctree(f)
    if ctree[0] != T_RETURN:
        raise NotImplementedError('must be a function.')

    r_expr = f2cl_expr(ctree[1], f.__globals__, f.__closure__)
    r_fname = cl_fname

    if cl_param_type is not None:
        tpl = ("static float {r_fname}(float t, {r_param_type} p) {{\n"
               "    return {r_expr};\n"
               "}}")
        return tpl.format(r_fname=r_fname, r_expr=r_expr, r_param_type=cl_param_type)

    tpl = ("static float {r_fname}(float t) {{\n"
           "    return {r_expr};\n"
           "}}")
    return tpl.format(r_fname=r_fname, r_expr=r_expr)


def glob_attr_to_cl(mod, attr):
    """ returns cl representation of a global attribute, if possible.
        """
    if len(attr):
        return glob_attr_to_cl(getattr(mod, attr[0]), attr[1:])

    if isinstance(mod, (int, np.int32)):
        return r_clint(mod)

    if isinstance(mod, (float, np.float32, np.float64)):
        return r_clfloat(mod)

    if isinstance(mod, (np.complex64, np.complex128)):
        raise ValueError('complex numbers not supported in real function.')

    try:
        mod_cl = next(m for m in GLOBALS_MAP if m[0] is mod)
        return mod_cl[1]
    except StopIteration:
        raise RuntimeError('cannot translate {} into OpenCL, sryy'.format(mod))


def instruction_scalar(instruction):
    """ identifies scalar instruction
        """
    if not isinstance(instruction, dis.Instruction):
        return instruction

    if instruction.opname == "LOAD_CONST":
        return (T_VAL, instruction.argval)

    if instruction.opname == "LOAD_GLOBAL":
        return (T_GLOBAL_SYMBOLE, instruction.argval)

    if instruction.opname == "LOAD_METHOD":
        return (T_METHOD, instruction.argval)

    if instruction.opname == "LOAD_FAST":
        return (T_SYMBOLE, instruction.argval, instruction.arg)

    raise ValueError(instruction)


BIN_MAP = {
    'BINARY_ADD': '+',
    'BINARY_MULTIPLY': '*',
    'BINARY_POWER': '**',
    'BINARY_SUBTRACT': '-',
    'BINARY_TRUE_DIVIDE': '/',
    'BINARY_MODULO': '%',
}

def f2cl_expr(ctree, glb, closure):
    """ maps ctree to an expression
        """
    if ctree[0] == T_FUNC:
        fname = f2cl_expr(ctree[1], glb, closure)
        return "{}({})".format(fname, ', '.join([f2cl_expr(c, glb, closure) for c in ctree[2]]))

    if ctree[0] == T_SYMBOLE:
        if ctree[2] == 0:
            return 't'
        if ctree[2] == 1:
            return 'p'
        raise RuntimeError('to many local symbols')

    if ctree[0] == T_GLOBAL_SYMBOLE:
        glob_key = ctree[1].split('.')
        if not glob_key[0] in glb:
            raise RuntimeError('unkown global {}'.format(ctree[1]))
        return glob_attr_to_cl(glb[glob_key[0]], glob_key[1:])

    if ctree[0] == T_VAL:
        if isinstance(ctree[1], int):
            return r_clint(ctree[1])
        if isinstance(ctree[1], (float, np.double)):
            return r_clfloat(ctree[1])
        if isinstance(ctree[1], str):
            return "'{}'".format(ctree[1])
        raise NotImplementedError(str(ctree))

    if ctree[0] == T_DICT:
        if ctree[2][0] != T_VAL:
            msg = "only static subscription supported: {}[{}] given."
            raise NotImplementedError(msg.format(ctree[1], ctree[2]))
        elif not isinstance(ctree[2][1], str):
            msg = "only string subscription supported: {}[{}] given."
            raise NotImplementedError(msg.format(ctree[1], ctree[2]))
        return "{}.{}".format(f2cl_expr(ctree[1], {}, closure), ctree[2][1])

    if ctree[0] == T_BIN:
        if ctree[1] in ['*', '+', '/', '-']:
            return "({} {} {})".format(
                f2cl_expr(ctree[2][0], glb, closure),
                ctree[1],
                f2cl_expr(ctree[2][1], glb, closure))
        raise NotImplementedError()

    if ctree[0] == T_DEREF:
        val = closure[ctree[1]].cell_contents
        if isinstance(val, int):
            return r_clint(val)
        if isinstance(val, (float, np.double)):
            return r_clfloat(val)
        if isinstance(val, str):
            return "'{}'".format(val)
        raise NotImplementedError(str(val))

    if ctree[0] == T_UNARY_NEGATIVE:
        return '-{}'.format(f2cl_expr(ctree[1], glb, closure))

    raise NotImplementedError(str(ctree))


def ctree_print(a, l=0):
    """ for debuging purpose: visualizes
        a ctree in a nested multi line
        format. """
    if isinstance(a, dis.Instruction):
        print(l * ' ' + str(a))

    elif not isinstance(a, tuple):
        print('!?')

    elif a[0] == T_FUNC:
        print(l * ' ' + a[0] + "(" + str(a[1]) + ")")
        for b in a[2]:
            ctree_print(b, l+1)

    elif a[0] == T_BIN:
        print(l * ' ' + a[0] + '(' + a[1] + ')')
        for b in a[2]:
            ctree_print(b, l+1)

    elif a[0] == T_SYMBOLE:
        print(l * ' ' + a[0] + '(' + str(a[1]) + ')')

    elif a[0] == T_GLOBAL_SYMBOLE:
        print(l * ' ' + a[0] + '(' + str(a[1]) + ')')

    elif a[0] == T_VAL:
        print(l * ' ' + a[0] + '(<' + str(a[1].__class__.__name__) + '>' + str(a[1]) + ')')

    elif a[0] == T_RETURN:
        print(l * ' ' + T_RETURN)
        ctree_print(a[1], l+1)


def create_ctree(f):
    """ creates some pseudo tree-like-thing from function
        instructions. This function only works on a small
        subset of functions. Currently only functions of
        single expressions (lambda), witout any branches,
        are supported.
        """
    read = []
    current = None

    for a in dis.get_instructions(f):
        if a.opname in ['STORE_FAST']:
            # XXX
            # - solve type problems: using type constraints it might be possible
            #   to map store expressions to valid OpenCL assignments.
            # - For now, only RETURN_VALUE is a known "routine"
            raise NotImplementedError('Only one line expressions are allowed. Sryy')

        elif a.opname in ['POP_JUMP_IF_FALSE']:
            # XXX
            # - solve branching. Currently _this_ reading process is quiet
            #   primitive which could make it messy to solve the problem at
            #   the moment.
            raise NotImplementedError('Only one line expressions are allowed. Sryy')

        elif a.opname in ['LOAD_CONST', 'LOAD_FAST', 'LOAD_GLOBAL', 'LOAD_METHOD']:
            read.append(instruction_scalar(a))

        elif a.opname == 'LOAD_ATTR':
            read[-1] = (read[-1][0], read[-1][1] + '.' + a.argval)

        elif a.opname == 'BINARY_SUBSCR':
            current = (T_DICT, read[-2], instruction_scalar(read[-1]))
            read = read[:-2]
            read.append(current)

        elif a.opname == 'COMPARE_OP':
            # XXX
            # - this opname is not needed as long as the IF/ELSE does
            #   not work.
            raise NotImplementedError('Waiting for if/else/while. Sryy')

        elif a.opname in BIN_MAP:
            current = (T_BIN, BIN_MAP[a.opname], [
                instruction_scalar(read[-2]),
                instruction_scalar(read[-1])])
            read = read[:-2]
            read.append(current)

        elif a.opname == 'CALL_FUNCTION':
            fargs = read[-a.arg:]
            fname = read[-a.arg-1]
            current = (T_FUNC, fname, [instruction_scalar(fa) for fa in fargs])
            read = read[:-a.arg-1]
            read.append(current)

        # python 3.7
        elif a.opname == 'CALL_METHOD':
            fargs = read[-a.arg:]
            if read[-a.arg-2][0] == T_GLOBAL_SYMBOLE:
                fname = (T_GLOBAL_SYMBOLE, read[-a.arg-2][1] + '.' + read[-a.arg-1][1])
                current = (T_FUNC, fname, [instruction_scalar(fa) for fa in fargs])
                read = read[:-a.arg-2]
                read.append(current)
            else:
                fname = read[-a.arg-1]
                current = (T_FUNC, fname, [instruction_scalar(fa) for fa in fargs])
                read = read[:-a.arg-1]
                read.append(current)

        elif a.opname == 'RETURN_VALUE':
            if len(read) != 1:
                for b in read:
                    print(b)
                raise RuntimeError('do not know how to deal with this?!')
            current = (T_RETURN, read[-1])
            read.append(current)

        elif a.opname == 'LOAD_DEREF':
            current = (T_DEREF, a.arg, a.argrepr)
            read.append(current)

        elif a.opname == 'UNARY_NEGATIVE':
            current = (T_UNARY_NEGATIVE, read[-1])
            read = read[:-1]
            read.append(current)

        elif a.opname == 'UNARY_POSITIVE':
            # + has no affect so we ignore it.
            pass

        else:
            raise RuntimeError('did not understand?! {}'.format(a))

    return current
