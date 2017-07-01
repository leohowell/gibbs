# -*- coding: utf-8 -*-

import dis
import sys
import logging
import operator

from .objects import Function, Block, Finished, BlockType
from .errors import InstructionNotFound, VirtualMachineError


logger = logging.getLogger(__name__)


class ISA(object):
    """Instruction Set Architecture

    class dis.Instruction
        opname: human readable name for operation
        opcode: numeric code for operation
        arg: numeric argument to operation (if any), otherwise None
        argval: resolved arg value (if known), otherwise same as arg
        argrepr: human readable description of operation argument
        offset: start index of operation within bytecode sequence
        starts_line: line started by this opcode (if any), otherwise None
        is_jump_target: True if other code jumps to here, otherwise False
    """

    def __init__(self, frame, vm):
        self.frame = frame
        self.vm = vm

    def execute(self, instruction):
        """
        :type instruction: dis.Instruction
        """
        op = instruction.opname.lower()

        try:
            if op.startswith('binary'):
                fn = self._binary_operation
            elif op.startswith('inplace'):
                fn = self._inplace_operation
            else:
                fn = getattr(self, op)
        except AttributeError:
            raise InstructionNotFound('Unknown instruction: <{}>'
                                      .format(instruction.opname))

        try:
            result = fn(instruction)
            result = result or Finished.SUCCEED
        except Exception as err:
            logger.exception('Exception happened: {}'.format(repr(err)))
            self.frame.f_exception = sys.exc_info()[:2] + (None, )
            result = Finished.EXCEPTION

        return result

    def top(self):
        return self.frame.stack[-1]

    def peek(self, n):
        return self.frame.stack[-n]

    def push(self, *values):
        self.frame.stack.extend(values)

    def pop(self, n=0):
        return self.frame.stack.pop(-(n+1))

    def popn(self, n):
        if not n:
            return []
        values = self.frame.stack[-n:]
        self.frame.stack[-n:] = []
        return tuple(values)

    def jump(self, offset):
        self.frame.f_lasti = offset

    def push_frame_block(self, _type, handler=None, level=None):
        if level is None:
            level = len(self.frame.stack)
        self.frame.block_stack.append(Block(_type, handler, level))

    def pop_frame_block(self):
        return self.frame.block_stack.pop()

    def unwind_frame_block(self, block):
        if block.type == BlockType.EXCEPT_HANDLER:
            offset = 2
        else:
            offset = 0

        while len(self.frame.stack) > block.level + offset:
            self.pop()

        if offset == 2:
            tb, value, exctype = self.popn(3)
            self.frame.f_exception = exctype, value, tb

    ########################
    # General instructions #
    ########################

    def nop(self, _):
        """
        Do nothing code.
        Used as a placeholder by the bytecode optimizer.
        """

    def pop_top(self, _):
        """Removes the top-of-stack (TOS) item."""
        self.pop()

    def rot_two(self, _):
        """Swaps the two top-most stack items."""
        x, y = self.popn(2)
        self.push(y, x)

    def rot_three(self, _):
        """
        Lifts second and third stack item one position up,
        moves top down to position three.
        """
        x, y, z = self.popn(3)
        self.push(z, x, y)

    def dup_top(self, _):
        """Duplicates the reference on top of the stack."""
        self.push(self.top())

    def dup_top_two(self, _):
        """
        Duplicates the two references on top of the stack,
        leaving them in the same order.
        """
        x, y = self.popn(2)
        self.push(x, y, x, y)

    ####################
    # Unary operations #
    ####################

    def unary_positive(self, _):
        """Implements TOS = +TOS."""

    def unary_negative(self, _):
        """Implements TOS = -TOS."""
        self.push(-self.pop())

    def unary_not(self, _):
        """Implements TOS = not TOS."""
        self.push(not self.pop())

    def unary_invert(self, _):
        """Implements TOS = ~TOS."""
        self.push(~self.pop())

    def get_iter(self, _):
        """Implements TOS = iter(TOS)."""
        self.push(iter(self.pop()))

    def get_yield_from_iter(self, instr):
        """
        If TOS is a generator iterator or coroutine object
        it is left as is. Otherwise, implements TOS = iter(TOS).
        """
        raise NotImplementedError

    #####################
    # Binary operations #
    #####################

    BINARY_OPERATORS = {
        'MATRIX_MULTIPLY': operator.matmul,
        'POWER': operator.pow,
        'MULTIPLY': operator.mul,
        'MODULO': operator.mod,
        'ADD': operator.add,
        'SUBTRACT': operator.sub,
        'SUBSCR': operator.getitem,
        'FLOOR_DIVIDE': operator.floordiv,
        'TRUE_DIVIDE': operator.truediv,
        'LSHIFT': operator.lshift,
        'RSHIFT': operator.rshift,
        'AND': operator.and_,
        'XOR': operator.xor,
        'OR': operator.or_,
    }

    def _binary_operation(self, instr):
        op = instr.opname.split('_', 1)[1]
        x, y = self.popn(2)
        result = self.BINARY_OPERATORS[op](x, y)
        self.push(result)

    #######################
    # In-place operations #
    #######################

    INPLACE_OPERATORS = {
        'MATRIX_MULTIPLY': operator.imatmul,
        'FLOOR_DIVIDE': operator.ifloordiv,
        'TRUE_DIVIDE': operator.itruediv,
        'ADD': operator.iadd,
        'SUBTRACT': operator.isub,
        'MULTIPLY': operator.imul,
        'MODULO': operator.imod,
        'POWER': operator.ipow,
        'LSHIFT': operator.ilshift,
        'RSHIFT': operator.irshift,
        'AND': operator.iand,
        'XOR': operator.ixor,
        'OR': operator.ior,
    }

    def _inplace_operation(self, instr):
        op = instr.opname.split('_', 1)[1]
        x, y = self.popn(2)
        result = self.INPLACE_OPERATORS[op](x, y)
        self.push(result)

    def store_subscr(self, _):
        """Implements TOS1[TOS] = TOS2."""
        value, obj, item = self.popn(3)
        obj[item] = value

    def delete_subscr(self, _):
        """Implements del TOS1[TOS]."""
        obj, item = self.popn(2)
        del obj[item]

    #####################
    # Coroutine opcodes #
    #####################

    #########################
    # Miscellaneous opcodes #
    #########################

    def print_expr(self, _):
        """
        Implements the expression statement for the interactive mode.
        TOS is removed from the stack and printed.
        In non-interactive mode, an expression statement is terminated
        with POP_TOP.
        """
        print(self.pop())

    def break_loop(self, _):
        """Terminates a loop due to a break statement."""
        return Finished.BREAK

    def continue_loop(self, instr):
        """Continues a loop due to a continue statement."""
        raise RuntimeError

    # For all of the SET_ADD, LIST_APPEND and MAP_ADD instructions,
    # while the added value or key/value pair is popped off,
    # the container object remains on the stack so that it is available
    # for further iterations of the loop.

    def set_add(self, instr):
        """
        Calls set.add(TOS1[-i], TOS).
        Used to implement set comprehensions.
        """
        value = self.pop()
        _set = self.peek(instr.argval)
        _set.add(value)

    def list_append(self, instr):
        """
        Calls list.append(TOS[-i], TOS).
        Used to implement list comprehensions.
        """
        value = self.pop()
        _list = self.peek(instr.argval)
        _list.append(value)

    def map_add(self, instr):
        """
        Calls dict.setitem(TOS1[-i], TOS, TOS1).
        Used to implement dict comprehensions.
        """
        value, key = self.popn(2)
        _dict = self.peek(instr.argval)
        _dict[key] = value

    def return_value(self, _):
        """Returns with TOS to the caller of the function."""
        self.frame.f_result = self.pop()
        if self.frame.generator:
            self.frame.generator.finished = True
        return Finished.RETURN

    def yield_value(self, instr):
        """Pops TOS and yields it from a generator."""
        self.frame.f_result = self.pop()
        return Finished.YIELD

    def yield_from(self, instr):
        """Pops TOS and delegates to it as a subiterator from a generator."""
        raise NotImplementedError

    def setup_annotations(self, instr):
        """
        Checks whether __annotations__ is defined in locals(), if not it
        is set up to an empty dict.
        This opcode is only emitted if a class or module body contains
        variable annotations statically.
        """
        raise NotImplementedError

    def import_star(self, _):
        """
        Loads all symbols not starting with '_' directly from the module
        TOS to the local namespace. The module is popped after loading all
        names. This opcode implements from module import *.
        """
        mod = self.pop()
        if getattr(mod, '__all__', None):
            attrs = mod.__all__
        else:
            attrs = dir(mod)

        for attr in attrs:
            if not attr.startswith('_'):
                self.frame.f_locals[attr] = getattr(mod, attr)

    def pop_block(self, _):
        """
        Removes one block from the block stack.

        Per frame, there is a stack of blocks, denoting nested loops,
        try statements, and such.
        """
        self.frame.block_stack.pop()

    def pop_except(self, _):
        """
        Removes one block from the block stack.

        The popped block must be an exception handler block, as implicitly
        created when entering an except handler. In addition to popping
        extraneous values from the frame stack, the last three popped
        values are used to restore the exception state.
        """
        raise NotImplementedError

    def end_finally(self, _):
        """
        Terminates a finally clause.

        The interpreter recalls whether the exception has to be re-raised,
        or whether the function returns, and continues with the outer-next
        block.
        """
        raise NotImplementedError

    def load_build_class(self, _):
        """
        Pushes builtins.__build_class__() onto the stack.

        It is later called by CALL_FUNCTION to construct a class.
        """
        self.push(self.frame.f_globals['__builtins__']['__build_class__'])

    def setup_with(self, instr):
        """
        This opcode performs several operations before a with block starts.

        First, it loads __exit__() from the context manager and pushes it
        onto the stack for later use by WITH_CLEANUP.
        Then, __enter__() is called, and a finally block pointing to delta
        is pushed.
        Finally, the result of calling the enter method is pushed onto the
        stack.
        The next opcode will either ignore it (POP_TOP), or store it in (a)
        variable(s) (STORE_FAST, STORE_NAME, or UNPACK_SEQUENCE).
        """
        raise NotImplementedError

    def with_cleanup_start(self, _):
        """
        Cleans up the stack when a with statement block exits.

        TOS is the context manager’s __exit__() bound method.
        Below TOS are 1–3 values indicating how/why the finally clause
        was entered:
            SECOND = None
            (SECOND, THIRD) = (WHY_{RETURN,CONTINUE}), retval
            SECOND = WHY_*; no retval below it
            (SECOND, THIRD, FOURTH) = exc_info()
        In the last case, TOS(SECOND, THIRD, FOURTH) is called,
        otherwise TOS(None, None, None). Pushes SECOND and result of the
        call to the stack.
        """
        raise NotImplementedError

    def with_cleanup_finish(self, _):
        """
        Pops exception type and result of ‘exit’ function call from the stack.

        If the stack represents an exception, and the function call returns
        a ‘true’ value, this information is “zapped” and replaced with a
        single WHY_SILENCED to prevent END_FINALLY from re-raising the
        exception. (But non-local gotos will still be resumed.)
        """
        raise NotImplementedError

    def store_name(self, instr):
        """Implements name = TOS.

        namei is the index of name in the attribute co_names of the code
        object. The compiler tries to use STORE_FAST or STORE_GLOBAL
        if possible.
        """
        self.frame.f_locals[instr.argval] = self.pop()

    def delete_name(self, instr):
        """
        Implements del name,

        where namei is the index into co_names attribute of the code object.
        """
        del self.frame.f_locals[instr.argval]

    def unpack_sequence(self, _):
        """
        Unpacks TOS into count individual values, which are put onto the
        stack right-to-left.
        """
        seq = self.pop()
        for x in reversed(seq):
            self.push(x)

    def unpack_ex(self, instr):
        """
        Implements assignment with a starred target: Unpacks an iterable
        in TOS into individual values, where the total number of values
        can be smaller than the number of items in the iterable: one of
        the new values will be a list of all leftover items.

        The low byte of counts is the number of values before the list
        value, the high byte of counts the number of values after it.
        The resulting values are put onto the stack right-to-left.
        """
        raise NotImplementedError

    def store_attr(self, instr):
        """
        Implements TOS.name = TOS1, where namei is the index of name in
        co_names.
        """
        value, _object = self.popn(2)
        setattr(_object, instr.argval, value)

    def delete_attr(self, instr):
        """Implements del TOS.name, using namei as index into co_names."""
        _object = self.pop()
        delattr(_object, instr.argval)

    def store_global(self, instr):
        """Works as STORE_NAME, but stores the name as a global."""
        self.frame.f_globals[instr.argval] = self.pop()

    def delete_global(self, instr):
        """Works as DELETE_NAME, but deletes a global name."""
        raise NotImplementedError

    def load_const(self, instr):
        """Pushes co_consts[consti] onto the stack."""
        self.push(instr.argval)

    def _load_var(self, name, scopes):
        for scope in scopes:
            if name in scope:
                self.push(scope[name])
                return
        else:
            raise NameError('Name: {} is not defined'.format(name))

    def load_name(self, instr):
        """Pushes the value associated with co_names[namei] onto the stack."""
        f = self.frame
        self._load_var(instr.argval, (f.f_locals, f.f_globals, f.f_builtins))

    def build_tuple(self, instr):
        """
        Creates a tuple consuming count items from the stack,
        and pushes the resulting tuple onto the stack.
        """
        _tuple = self.popn(instr.argval)
        self.push(tuple(_tuple))

    def build_list(self, instr):
        """Works as BUILD_TUPLE, but creates a list."""
        _list = self.popn(instr.argval)
        self.push(list(_list))

    def build_set(self, instr):
        """Works as BUILD_TUPLE, but creates a set."""
        _set = self.popn(instr.argval)
        self.push(set(_set))

    def build_map(self, instr):
        """
        Pushes a new dictionary object onto the stack.

        Pops 2 * count items so that the dictionary holds count
        entries: {..., TOS3: TOS2, TOS1: TOS}.
        """
        values = self.popn(instr.argval * 2)
        i = iter(values)
        self.push(dict(zip(i, i)))

    def build_const_key_map(self, instr):
        """
        The version of BUILD_MAP specialized for constant keys.

        count values are consumed from the stack. The top element on
        the stack contains a tuple of keys.
        """
        keys = self.pop()
        values = self.popn(len(keys))
        self.push(dict(zip(keys, values)))

    def build_string(self, instr):
        """
        Concatenates count strings from the stack and pushes the
        resulting string onto the stack.
        """
        raise NotImplementedError

    def build_tuple_unpack(self, instr):
        """
        Pops count iterables from the stack, joins them in a single
        tuple, and pushes the result. Implements iterable unpacking
        in tuple displays (*x, *y, *z).
        """
        raise NotImplementedError

    def build_tuple_unpack_with_call(self, instr):
        """
        This is similar to BUILD_TUPLE_UNPACK, but is used for
        f(*x, *y, *z) call syntax. The stack item at position count + 1
        should be the corresponding callable f.
        """
        raise NotImplementedError

    def build_list_unpack(self, instr):
        """
        This is similar to BUILD_TUPLE_UNPACK, but pushes a list instead
        of tuple. Implements iterable unpacking in list displays [*x, *y, *z].
        """
        raise NotImplementedError

    def build_set_unpack(self, instr):
        """
        This is similar to BUILD_TUPLE_UNPACK, but pushes a set instead
        of tuple. Implements iterable unpacking in set displays {*x, *y, *z}.
        """
        raise NotImplementedError

    def build_map_unpack(self, instr):
        """
        Pops count mappings from the stack, merges them into a single
        dictionary, and pushes the result. Implements dictionary unpacking
        in dictionary displays {**x, **y, **z}.
        """
        raise NotImplementedError

    def build_map_unpack_with_call(self, instr):
        """
        This is similar to BUILD_MAP_UNPACK, but is used for
        f(**x, **y, **z) call syntax. The stack item at position
        count + 2 should be the corresponding callable f.
        """
        raise NotImplementedError

    def load_attr(self, instr):
        """Replaces TOS with getattr(TOS, co_names[namei])."""
        tos = self.pop()
        self.push(getattr(tos, instr.argval))

    # from dis.cmp_op
    COMPARE_OPERATORS = {
        '<': operator.lt,
        '<=': operator.le,
        '==': operator.eq,
        '!=': operator.ne,
        '>': operator.gt,
        '>=': operator.ge,
        'in': lambda x, y: x in y,
        'not in': lambda x, y: x not in y,
        'is': operator.is_,
        'is not': operator.is_not,
        'exception match':
            lambda x, y: issubclass(x, Exception) and issubclass(x, y),
    }

    def compare_op(self, instr):
        """
        Performs a Boolean operation.

        The operation name can be found in cmp_op[opname].
        """
        x, y = self.popn(2)
        result = self.COMPARE_OPERATORS[instr.argval](x, y)
        self.push(result)

    def import_name(self, instr):
        """
        Imports the module co_names[namei].
        TOS and TOS1 are popped and provide the fromlist and level
         arguments of __import__().
        The module object is pushed onto the stack.
        The current namespace is not affected: for a proper import statement,
         a subsequent STORE_FAST instruction modifies the namespace.
        """
        level, fromlist = self.popn(2)
        f = self.frame
        _import = __import__(instr.argval, f.f_globals, f.f_locals,
                             fromlist, level)
        self.push(_import)

    def import_from(self, instr):
        """
        Loads the attribute co_names[namei] from the module found in TOS.

        The resulting object is pushed onto the stack, to be subsequently
        stored by a STORE_FAST instruction.
        """
        _module = self.top()
        self.push(getattr(_module, instr.argval))

    def jump_forward(self, instr):
        """Increments bytecode counter by delta."""
        self.jump(instr.argval)

    def pop_jump_if_true(self, instr):
        """
        If TOS is true, sets the bytecode counter to target. TOS is popped.
        """
        value = self.pop()
        if value:
            self.jump(instr.argval)

    def pop_jump_if_false(self, instr):
        """
        If TOS is false, sets the bytecode counter to target. TOS is popped.
        """
        value = self.pop()
        if not value:
            self.jump(instr.argval)

    def jump_if_true_or_pop(self, instr):
        """
        If TOS is true, sets the bytecode counter to target and leaves
        TOS on the stack. Otherwise (TOS is false), TOS is popped.
        """
        value = self.top()
        if value:
            self.jump(instr.argval)
        else:
            self.pop()

    def jump_if_false_or_pop(self, instr):
        """
        If TOS is false, sets the bytecode counter to target and leaves
        TOS on the stack. Otherwise (TOS is true), TOS is popped.
        """
        value = self.top()
        if not value:
            self.jump(instr.argval)
        else:
            self.pop()

    def jump_absolute(self, instr):
        """Set bytecode counter to target."""
        self.jump(instr.argval)

    def for_iter(self, instr):
        """
        TOS is an iterator. Call its __next__() method.

        If this yields a new value, push it on the stack (leaving the
        iterator below it). If the iterator indicates it is exhausted
        TOS is popped, and the byte code counter is incremented by delta.
        """
        iter_obj = self.top()
        try:
            v = next(iter_obj)
            self.push(v)
        except StopIteration:
            self.pop()
            self.jump(instr.argval)

    def load_global(self, instr):
        """Loads the global named co_names[namei] onto the stack."""
        f = self.frame
        self._load_var(instr.argval, (f.f_globals, f.f_builtins))

    def setup_loop(self, instr):
        """
        Pushes a block for a loop onto the block stack.

        The block spans from the current instruction with a size of
        delta bytes.
        """
        self.push_frame_block(BlockType.LOOP, instr.argval)

    def setup_except(self, instr):
        """
        Pushes a try block from a try-except clause onto the block stack.
        delta points to the first except block.
        """
        self.push_frame_block(BlockType.SETUP_EXCEPT, instr.argval)

    def load_fast(self, instr):
        """
        Pushes a reference to the local co_varnames[var_num] onto the stack.

        CPython implementation detail:
            CPython generates implicit parameter names of the form .0
            on the code objects used to implement comprehensions and
            generator expressions.

        Changed in version 3.6:
            These parameter names are exposed by this module as names
            like implicit0.

        See: https://docs.python.org/3/library/inspect.html#inspect.Parameter
        """
        f = self.frame

        if instr.argval == '.0':
            self.push(f.f_locals.get('implicit0'))
            return

        self._load_var(instr.argval, (f.f_locals,))

    def store_fast(self, instr):
        """Stores TOS into the local co_varnames[var_num]."""
        self.frame.f_locals[instr.argval] = self.pop()

    def delete_fast(self, instr):
        """Deletes local co_varnames[var_num]."""
        del self.frame.f_locals[instr.argval]

    def store_annotation(self, instr):
        """Stores TOS as locals()['__annotations__'][co_names[namei]] = TOS."""
        raise NotImplementedError

    def load_closure(self, instr):
        """
        Pushes a reference to the cell contained in slot i of the cell
        and free variable storage. The name of the variable is
        co_cellvars[i] if i is less than the length of co_cellvars.
        Otherwise it is co_freevars[i - len(co_cellvars)].
        """
        self.push(self.frame.cells[instr.argval])

    def load_deref(self, instr):
        """
        Loads the cell contained in slot i of the cell and free variable
        storage. Pushes a reference to the object the cell contains on
        the stack.
        """
        self.push(self.frame.cells[instr.argval].get())

    def load_classderef(self, instr):
        """
        Much like LOAD_DEREF but first checks the locals dictionary
        before consulting the cell. This is used for loading free
        variables in class bodies.
        """
        raise NotImplementedError

    def store_deref(self, instr):
        """
        Stores TOS into the cell contained in slot i of the cell and
        free variable storage.
        """
        raise NotImplementedError

    def delete_deref(self, instr):
        """
        Empties the cell contained in slot i of the cell and free
        variable storage. Used by the del statement.
        """
        raise NotImplementedError

    def _do_raise(self, exc, cause):
        pass

    def raise_varargs(self, instr):
        """
        Raises an exception. argc indicates the number of parameters
        to the raise statement, ranging from 0 to 3. The handler will
        find the traceback as TOS2, the parameter as TOS1, and the
        exception as TOS.
        """
        cause = exc = None
        if instr.argval == 2:
            cause = self.pop()
            exc = self.pop()
        elif instr.argval == 1:
            exc = self.pop()

        return self._do_raise(exc, cause)

    def call_function(self, instr):
        """
        Calls a function.
        argc indicates the number of positional arguments.
        The positional arguments are on the stack, with the right-most
        argument on top. Below the arguments, the function object to
        call is on the stack. Pops all function arguments, and the
        function itself off the stack, and pushes the return value.

        This opcode is used only for calls with positional arguments.
        """
        args = self.popn(instr.argval)
        fn = self.pop()

        # FIXME: make Function a real func object
        if fn == __build_class__ and getattr(args[0], 'real_func', None):
            args = list(args)
            args[0] = args[0].real_func

        result = fn(*args)
        self.push(result)

    def call_function_kw(self, instr):
        """
        Calls a function.
        argc indicates the number of arguments (positional and keyword).
        The top element on the stack contains a tuple of keyword argument
        names. Below the tuple, keyword arguments are on the stack, in
        the order corresponding to the tuple. Below the keyword arguments,
        the positional arguments are on the stack, with the right-most
        parameter on top. Below the arguments, the function object to call
        is on the stack. Pops all function arguments, and the function
        itself off the stack, and pushes the return value.

        Keyword arguments are packed in a tuple instead of a dictionary,
        argc indicates the total number of arguments
        """
        keys = self.pop()
        values = self.popn(len(keys))

        args = self.popn(instr.argval - len(keys))
        kwargs = dict(zip(keys, values))

        fn = self.pop()
        result = fn(*args, **kwargs)
        self.push(result)

    def call_function_ex(self, instr):
        """
        Calls a function.
        The lowest bit of flags indicates whether the var-keyword argument
        is placed at the top of the stack. Below the var-keyword argument,
        the var-positional argument is on the stack. Below the arguments,
        the function object to call is placed.

        Pops all function arguments, and the function itself off the stack,
        and pushes the return value. Note that this opcode pops at most three
        items from the stack. Var-positional and var-keyword arguments are
        packed by BUILD_MAP_UNPACK_WITH_CALL and BUILD_MAP_UNPACK_WITH_CALL.
        """
        raise NotImplementedError

    _POS_ARG = 0x01
    _KW_ONLY_ARG = 0x02
    _ANNOTATION = 0x04
    _CELL_TUPLE = 0x08

    def make_function(self, instr):
        """
        Pushes a new function object on the stack.

        From bottom to top, the consumed stack must consist of values
        if the argument carries a specified flag value

            0x01 a tuple of default argument objects in positional order
            0x02 a dictionary of keyword-only parameters’ default values
            0x04 an annotation dictionary
            0x08 a tuple containing cells for free variables, making a closure
            the code associated with the function (at TOS1)
            the qualified name of the function (at TOS)

        """
        name = self.pop()
        code = self.pop()

        closure = None
        defaults = None
        kwdefaults = None
        annotations = {}

        flag = instr.argval
        if flag & self._CELL_TUPLE:
            closure = self.pop()

        if flag & self._ANNOTATION:
            annotations = self.pop()

        if flag & self._KW_ONLY_ARG:
            kwdefaults = self.pop()

        if flag & self._POS_ARG:
            defaults = self.pop()

        _globals = self.frame.f_globals
        fn = Function(name, code, _globals, defaults, kwdefaults, annotations,
                      closure, self.vm)
        self.push(fn)

    def build_slice(self, instr):
        """
        Pushes a slice object on the stack. argc must be 2 or 3.
        If it is 2, slice(TOS1, TOS) is pushed; if it is 3,
        slice(TOS2, TOS1, TOS) is pushed. See the slice() built-in
        function for more information.
        """
        if instr.argval == 2:
            tos1, tos = self.popn(2)
            self.push(slice(tos1, tos))
        elif instr.argval == 3:
            tos2, tos1, tos = self.popn(3)
            self.push(slice(tos2, tos1, tos))
        else:
            raise VirtualMachineError('Got build bad slice parameter: {}'
                                      .format(instr.argval))

    def extended_arg(self, instr):
        """
        Prefixes any opcode which has an argument too big to fit into
        the default two bytes. ext holds two additional bytes which,
        taken together with the subsequent opcode’s argument, comprise
        a four-byte argument, ext being the two most-significant bytes.
        """

    def format_value(self, instr):
        """
        Used for implementing formatted literal strings (f-strings).
        Pops an optional fmt_spec from the stack, then a required value.
        flags is interpreted as follows:

            (flags & 0x03) == 0x00: value is formatted as-is.
            (flags & 0x03) == 0x01: call str() on value before formatting it.
            (flags & 0x03) == 0x02: call repr() on value before formatting it.
            (flags & 0x03) == 0x03: call ascii() on value before formatting it.
            (flags & 0x04) == 0x04: pop fmt_spec from the stack and use it,
                                    else use an empty fmt_spec.
        Formatting is performed using PyObject_Format(). The result is
        pushed on the stack.
        """
        raise NotImplementedError

    def have_argument(self, instr):
        """
        This is not really an opcode.
        It identifies the dividing line between opcodes which don’t use
        their argument and those that do
        (< HAVE_ARGUMENT and >= HAVE_ARGUMENT, respectively).
        """
        raise NotImplementedError
