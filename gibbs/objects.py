# -*- coding: utf-8 -*-

import enum
import types
import inspect
import logging
import collections

logger = logging.getLogger(__name__)


class Finished(enum.Enum):
    SUCCEED = enum.auto()
    EXCEPTION = enum.auto()
    YIELD = enum.auto()
    RERAISE = enum.auto()
    RETURN = enum.auto()
    BREAK = enum.auto()
    CONTINUE = enum.auto()


class BlockType(enum.Enum):
    LOOP = enum.auto()
    FINALLY = enum.auto()
    EXCEPT_HANDLER = enum.auto()
    SETUP_EXCEPT = enum.auto()


Block = collections.namedtuple('Block', ['type', 'handler', 'level'])


class Cell(object):
    def __init__(self, value):
        self.contents = value

    def get(self):
        return self.contents

    def set(self, value):
        self.contents = value


class Frame(object):
    def __init__(self, f_code, f_globals, f_locals, f_back):
        """
        :param f_code: code object being executed in this frame
        :param f_globals: global namespace seen by this frame
        :param f_locals: local namespace seen by this frame
        :param f_back: next outer frame object (this frame’s caller)

        :local f_builtins: builtins namespace seen by this frame
        :local f_lineno: current line number in Python source code
        :local f_lasti: index of last attempted instruction in bytecode
        """

        self.f_code = f_code
        self.f_globals = f_globals
        self.f_locals = f_locals
        self.f_back = f_back

        self.stack = []
        self.f_result = None
        self.f_exception = None

        if f_back:
            self.f_builtins = f_back.f_builtins
        else:
            self.f_builtins = f_locals['__builtins__']
            if hasattr(self.f_builtins, '__dict__'):
                self.f_builtins = self.f_builtins.__dict__

        self.f_lineno = f_code.co_firstlineno
        self.f_lasti = 0

        # co_cellvars：local variables referenced by nested functions
        if f_code.co_cellvars:
            self.cells = {}
            f_back.cells = f_back.cells or {}
            for var in f_code.co_cellvars:
                cell = Cell(self.f_locals.get(var))
                f_back.cells[var] = self.cells[var] = cell
        else:
            self.cells = None

        # co_freevars: closure variable referenced by current function
        if f_code.co_freevars:
            self.cells = self.cells or {}
            for var in f_code.co_freevars:
                assert f_back.cells, 'f_back.cells: {}'.format(f_back.cells)
                self.cells[var] = f_back.cells[var]

        self.block_stack = []
        self.generator = None

    def __repr__(self):
        return '<Frame at: {} {} @ {}>'.format(id(self),
                                               self.f_code.co_filename,
                                               self.f_lineno)


class Method(object):
    def __init__(self, instance, _class, func):
        self.__self__ = instance
        self.im_class = _class
        self.__func__ = func

    def __repr__(self):
        name = '{}.{}'.format(self.im_class.__name__, self.__func__.__name__)
        if self.__self__ is None:
            return '<function method {}>'.format(name)
        else:
            return '<bound method {} of {}>'.format(name, self.__self__)

    def __call__(self, *args, **kwargs):
        if self.__self__ is None:
            return self.__func__(self.__self__, *args, **kwargs)
        else:
            return self.__func__(*args, **kwargs)


class Function(object):
    __slots__ = [
        '__code__', '__name__', '__defaults__', '__globals__',
        '__dict__', '__doc__', '__closure__',
        'vm', 'f_locals', 'real_func',
    ]

    def __init__(self, name, code, _globals, defaults, kwdefaults, annotations, closure, vm):
        """
        :param name: name with which this function was defined
        :param code: code object containing compiled function bytecode
        :param _globals: global namespace in which this function was defined
        :param defaults: tuple of any default values for positional or
                         keyword parameters
        :param closure: None or a tuple of cells that contain bindings
                        for the function’s free variables.
        :param vm: VirtualMachine instance
        """
        self.__code__ = code
        self.__name__ = name or code.co_name
        self.__defaults__ = defaults
        self.__kwdefaults__ = kwdefaults
        self.__globals__ = _globals
        self.__dict__ = {}
        self.__doc__ = code.co_consts[0] if code.co_consts else None
        self.__annotations__ = annotations
        if closure:
            self.__closure__ = tuple(self.make_cell(0) for _ in closure)
        else:
            self.__closure__ = closure

        self.vm = vm
        self.f_locals = self.vm.active_frame.f_locals

        # Build a Python func for use of inspect.getcallargs
        # because we conn't fake a <function> type
        self.real_func = types.FunctionType(code, _globals,
                                            closure=self.__closure__)
        self.real_func.__defaults__ = self.__defaults__
        self.real_func.__kwdefaults__ = kwdefaults
        self.real_func.__annotations__ = self.__annotations__

    def __repr__(self):
        return '<Function {} at {}>'.format(self.__name__, id(self))

    def __get__(self, instance, _class):
        if instance is not None:
            return Method(instance, _class, self)
        else:
            return self

    def __call__(self, *args, **kwargs):
        callargs = inspect.getcallargs(self.real_func, *args, **kwargs)

        frame = self.vm.make_frame(self.__code__, callargs,
                                   self.__globals__)

        # co_flags: an integer encoding a number of flags for the interpreter.
        # inspect.CO_VARARGS        4     for uses the *arguments syntax
        # inspect.CO_VARKEYWORDS    8     for uses the **keywords syntax
        # inspect.CO_GENERATOR      32    for function is a generator
        # Other bits in co_flags are reserved for internal use.

        if self.__code__.co_flags & inspect.CO_GENERATOR:
            gen = Generator(frame, self.vm)
            frame.generator = gen
            return gen

        return self.vm.run_frame(frame)

    @classmethod
    def make_cell(cls, value):
        fn = (lambda x: lambda: x)(value)
        return fn.__closure__[0]


class Generator(object):
    def __init__(self, gi_frame, vm):
        self.gi_frame = gi_frame
        self.vm = vm
        self.started = False
        self.finished = False

    def __iter__(self):
        return self

    def __next__(self):
        return self.send()

    def send(self, value=None):
        if not self.started and value is not None:
            raise TypeError("Can't send non-None value to a "
                            "just-started generator")
        self.gi_frame.stack.append(value)
        self.started = True
        result = self.vm.resume_frame(self.gi_frame)
        if self.finished:
            raise StopIteration(result)
        return result
