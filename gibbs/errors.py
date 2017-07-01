# -*- coding: utf-8 -*-


class VirtualMachineError(Exception):
    pass


class InstructionNotFound(VirtualMachineError):
    pass
