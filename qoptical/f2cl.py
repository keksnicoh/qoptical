# -*- coding: utf-8 -*-
""" module to dissassemble functions to generate
    OpenCL functions from it. Currently this module
    is capable of handling a small special case
    of functions:

        1. Single Expression
        2. No Conditionals
        3. 2 Arguments

    XXX
        - quiet experimental and messy module at the moment.

    :author: keksnicoh
"""

import dis

T_VAL = 'VAL'
T_VAR = 'VAR'
T_BINARY_ADD = 'ADD'
T_BINARY_ADD = 'ADD'
MODE_BIN = 1




import numpy as np
import math
GLOBALS_MAP = [
    (np.sin, 'sin', [(float, int)]),
    (np.cos, 'cos', [(float, int)]),
    (np.tan, 'tan', [(float, int)]),
    (np.arcsin, 'asin', [(float, int)]),
    (np.arccos, 'acos', [(float, int)]),
    (np.arctan, 'atan', [(float, int)]),
    (np.sinh, 'sinh', [(float, int)]),
    (np.cosh, 'cosh', [(float, int)]),
    (np.tanh, 'tanh', [(float, int)]),

    (math.sin, 'sin', [(float, int)]),
    (math.cos, 'cos', [(float, int)]),
    (math.tan, 'tan', [(float, int)]),
    (math.asin, 'asin', [(float, int)]),
    (math.acos, 'acos', [(float, int)]),
    (math.atan, 'atan', [(float, int)]),
    (math.sinh, 'sinh', [(float, int)]),
    (math.cosh, 'cosh', [(float, int)]),
    (math.tanh, 'tanh', [(float, int)]),
]

def r_clfloat(f, prec=None):
    """ renders an OpenCL float representation """
    if prec is not None:
        raise NotImplementedError()
    sf = str(f)
    return ''.join([sf, '' if '.' in sf else '.', 'f'])


def r_clint(f):
    """ renders an OpenCL integer representation """
    assert isinstance(f, int)
    return str(f)

def r_clfrac(p, q, prec=None):
    """ renders an OpenCL fractional representation """
    return "{}/{}".format(r_clfloat(p, prec), r_clfloat(q, prec))

def f2cl(f, cl_fname, cl_param_type=None):
    fglobals = f.__globals__
    ctree = create_ctree(f)
    if ctree[0] != 'T_RETURN':
        raise NotImplementedError('must be a function.')

    r_expr = f2cl_expr(ctree[1], f.__globals__)
    r_fname = cl_fname

    if cl_param_type is not None:
        tpl = ("static float {r_fname}(float t, {r_param_type} p) {{\n"
               "    return {r_expr};\n"
               "}}")
        return tpl.format(r_fname=r_fname, r_expr=r_expr, r_param_type=cl_param_type)
    else:
        tpl = ("static float {r_fname}(float t) {{\n"
               "    return {r_expr};\n"
               "}}")
        return tpl.format(r_fname=r_fname, r_expr=r_expr)

def glob_attr_to_cl(mod, attr):
    if len(attr):
        return glob_attr_to_cl(getattr(mod, attr[0]), attr[1:])

    try:
        mod_cl = next(m for m in GLOBALS_MAP if m[0] is mod)
        return mod_cl[1]
    except StopIteration:
        raise RuntimeError('cannot translate {} into OpenCL, sryy'.format(mod))

def instruction_scalar(instruction):
    if not isinstance(instruction, dis.Instruction):
        return instruction
    elif instruction.opname == "LOAD_CONST":
        return ('T_VAL', instruction.argval)
    elif instruction.opname == "LOAD_GLOBAL":
        return ('T_GLOBAL_SYMBOLE', instruction.argval)
    elif instruction.opname == "LOAD_FAST":
        return ('T_SYMBOLE', instruction.argval, instruction.arg)
    else:
        raise ValueError(instruction)
BIN_MAP = {
    'BINARY_ADD': '+',
    'BINARY_MULTIPLY': '*',
    'BINARY_POWER': '**',
    'BINARY_SUBTRACT': '-',
    'BINARY_TRUE_DIVIDE': '/',
    'BINARY_MODULO': '%',
}

def f2cl_expr(ctree, glb):

    if ctree[0] == 'T_FUNC':
        fname = f2cl_expr(ctree[1], glb)
        return "{}({})".format(fname, ', '.join([f2cl_expr(c, glb) for c in ctree[2]]))
    elif ctree[0] == 'T_SYMBOLE':
        if ctree[2] == 0:
            return 't'
        elif ctree[2] == 1:
            return 'p'
        else:
            raise RuntimeError('to many local symbols')
        return ctree[1]
    elif ctree[0] == 'T_GLOBAL_SYMBOLE':
        glob_key = ctree[1].split('.')
        if not glob_key[0] in glb:
            raise RuntimeError('unkown global {}'.format(ctree[1]))
        return glob_attr_to_cl(glb[glob_key[0]], glob_key[1:])
    elif ctree[0] == 'T_VAL':
        if isinstance(ctree[1], int) or isinstance(ctree[1], np.int):
            return r_clint(ctree[1])
        elif isinstance(ctree[1], float) or isinstance(ctree[1], np.float) \
          or isinstance(ctree[1], np.double):
            return r_clfloat(ctree[1])
        elif isinstance(ctree[1], str):
            return "'{}'".format(ctree[1])
        raise NotImplementedError(str(ctree))
    elif ctree[0] == 'T_DICT':
        if ctree[2][0] != 'T_VAL':
            msg = "only static subscription supported: {}[{}] given."
            raise NotImplementedError(msg.format(ctree[1], ctree[2]))
        elif not isinstance(ctree[2][1], str):
            msg = "only string subscription supported: {}[{}] given."
            raise NotImplementedError(msg.format(ctree[1], ctree[2]))
        return "{}.{}".format(f2cl_expr(ctree[1], {}), ctree[2][1])
    elif ctree[0] == 'T_BIN':
        if ctree[1] in ['*', '+', '/', '-']:
            return "({} {} {})".format(
                f2cl_expr(ctree[2][0], glb),
                ctree[1],
                f2cl_expr(ctree[2][1], glb))
        raise NotImplementedError()



    raise NotImplementedError(str(ctree))
def ctree_print(a, l=0):
    if isinstance(a, dis.Instruction):
        print(l * ' ' + str(a))
    elif not isinstance(a, tuple):
        return '!?'
    elif a[0] == 'T_FUNC':
        print(l * ' ' + a[0] + "(" + str(a[1]) + ")")
        for b in a[2]:
            ctree_print(b, l+1)
    elif a[0] == 'T_BIN':
        print(l * ' ' + a[0] + '(' + a[1] + ')')
        for b in a[2]:
            ctree_print(b, l+1)
    elif a[0] == 'T_SYMBOLE':
        print(l * ' ' + a[0] + '(' + str(a[1]) + ')')
    elif a[0] == 'T_GLOBAL_SYMBOLE':
        print(l * ' ' + a[0] + '(' + str(a[1]) + ')')
    elif a[0] == 'T_VAL':
        print(l * ' ' + a[0] + '(<' + str(a[1].__class__.__name__) + '>' + str(a[1]) + ')')
    elif a[0] == 'T_RETURN':
        print(l * ' ' + 'T_RETURN')
        ctree_print(a[1], l+1)

def create_ctree(f):
    mode = MODE_BIN
    args = []
    read = []
    current = None
    log = []

    for a in dis.get_instructions(f):
        log.append(a)
        read.append(a)
        if a.opname in ['STORE_FAST']:
            # XXX
            # - solve type problems: using type constraints it might be possible
            #   to map store expressions to valid OpenCL assignments.
            # - For now, only RETURN_VALUE is a known "routine"
            raise NotImplementedError('Only one line expressions are allowed. Sryy')
        if a.opname in ['POP_JUMP_IF_FALSE']:
            # XXX
            # - solve branching. Currently _this_ reading process is quiet
            #   primitive which could make it messy to solve the problem at
            #   the moment.
            raise NotImplementedError('Only one line expressions are allowed. Sryy')
        if a.opname in ['LOAD_CONST']:
            read[-1] = instruction_scalar(a)
        elif a.opname in ['LOAD_FAST']:
            read[-1] = instruction_scalar(a)
        elif a.opname in ['LOAD_GLOBAL']:
            read[-1] = instruction_scalar(a)
        elif a.opname == 'LOAD_ATTR':
            read[-2] = (read[-2][0], read[-2][1] + '.' + a.argval)
            read = read[:-1]
        elif a.opname == 'BINARY_SUBSCR':
            read[-3] = ('T_DICT', read[-3], instruction_scalar(read[-2]))
            read = read[:-2]
        elif a.opname == 'COMPARE_OP':
            # XXX
            # - this opname is not needed as long as the IF/ELSE does
            #   not work.
            raise NotImplementedError('Waiting for if/else/while. Sryy')
            current = ('T_COMPARE', a.argval, read[-3:-1])
            read = read[:-3]
            read.append(current)
        elif a.opname in BIN_MAP:
            current = ('T_BIN', BIN_MAP[a.opname], [
                instruction_scalar(read[-3]),
                instruction_scalar(read[-2])])
            read = read[:-3]
            read.append(current)
        elif a.opname == 'CALL_FUNCTION':
            fargs = read[-a.arg-1:-1]
            fname = read[-a.arg-2]
            current = ('T_FUNC', fname, [instruction_scalar(fa) for fa in fargs])
            read = read[:-a.arg-2]
            read.append(current)
        elif a.opname == 'RETURN_VALUE':
            if len(read) != 2:
                raise RuntimeError('do not know how to deal with this?!')
            current = ('T_RETURN', read[-2])
        else:
            raise RuntimeError('did not understand?!\n\n'+'\n'.join('>>> ' + str(l) for l in log))

    return current
