# -*- coding: utf-8 -*-

import dis
import logging
import textwrap

from .errors import VirtualMachineError
from .objects import Frame, Finished, BlockType
from .instruction import ISA


logger = logging.getLogger(__name__)


class VirtualMachine(object):
    def __init__(self):
        self.frames = []
        self.active_frame = None
        self.final_result = None
        self.last_exception = None

    def run_code(self, code, f_globals=None, f_locals=None):
        frame = self.make_frame(code, f_globals=f_globals, f_locals=f_locals)
        val = self.run_frame(frame)

        return val

    def push_frame(self, frame):
        self.frames.append(frame)
        self.active_frame = frame

    def pop_frame(self):
        self.frames.pop()
        self.active_frame = self.frames[-1] if self.frames else None

    def run_frame(self, frame):
        self.push_frame(frame)

        f = self.active_frame
        isa = ISA(f, self)

        instructions = {i.offset: i for i in dis.Bytecode(f.f_code)}
        while True:
            instr = instructions[f.f_lasti]

            f.f_lasti = instr.offset + 2
            status = isa.execute(instr)

            logger.debug('#{:<3}  {:<20}  {} {}'
                         .format(instr.offset, instr.opname,
                                 status.name, instr.argval))

            if status == Finished.EXCEPTION:
                raise frame.f_exception[0].with_traceback(frame.f_exception[1],
                                                          frame.f_exception[2])
            elif status == Finished.YIELD:
                break

            while status == Finished.BREAK and f.block_stack:
                status = self.run_block(status, isa)

            if status == Finished.RETURN:
                break

        self.final_result = f.f_result
        self.last_exception = f.f_exception
        self.pop_frame()
        return f.f_result

    def make_frame(self, code, callargs=None, f_globals=None, f_locals=None):
        if f_globals:
            f_locals = f_locals or f_globals
        elif self.frames:
            f_globals = self.active_frame.f_globals
            f_locals = {}
        else:
            f_globals = f_locals = {
                '__builtins__': __builtins__,
                '__name__': '__main__',
                '__doc__': None,
                '__package__': None,
            }

        if callargs:
            f_locals.update(callargs)

        return Frame(code, f_globals, f_locals, self.active_frame)

    def resume_frame(self, frame):
        frame.f_back = frame
        result = self.run_frame(frame)
        frame.f_back = None
        return result

    def run_block(self, status, isa):
        block = self.active_frame.block_stack[-1]
        if block.type == BlockType.LOOP and status == Finished.CONTINUE:
            isa.jump(self.final_result)
            return

        isa.pop_frame_block()
        isa.unwind_frame_block(block)

        if block.type == BlockType.LOOP and status == Finished.BREAK:
            isa.jump(block.handler)
            return Finished.SUCCEED

        if status == Finished.EXCEPTION and \
                block.type in (BlockType.EXCEPT_HANDLER, BlockType.FINALLY):
            pass
        elif block.type == BlockType.FINALLY:
            if status in (Finished.RETURN, Finished.CONTINUE):
                pass
        return status

    def text_to_code(self, text):
        _text = textwrap.dedent(text)
        return compile(_text, '<{}>'.format(id(self)), 'exec', 0, 1)
