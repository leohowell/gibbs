# gibbs

基于 Python 的 Python 解释器。

gibbs 项目的两个目标：
1. 了解statement, block, cell, frame, code等概念在Python中的实现。
2. 了解 函数、类、循环、异常、生成器、闭包、Coroutine等在Python中具体的指令及其实现。

仅适用于Python3.6及以上版本，Python2请查看 [byterun](https://github.com/nedbat/byterun) 项目。 

Python虚拟机指令在Python3.6中有大量不兼容变更，由于出于教学目的，gibbs项目不保持向前兼容，而仅兼容最新的稳定Python版本，同时也不提供任何性能保证。

目前gibbs项目中部分指令与测试用例尚未完成，欢迎提交PR

## 快速上手

```python
from gibbs.vm import VirtualMachine

text1 = '''
def foo(a, b):
    return a + b

print(foo(1, 99))
'''


vm = VirtualMachine()

code = vm.text_to_code(text1)

vm.run_code(code)
```

## 模块

1. `vm` Python虚拟机，负责frame调度。
2. `objects` Python运行时对象，提供堆栈。
3. `instruction` Python指令。
4. `errors` Python虚拟机异常模块。


## 致谢

- [byterun](http://aosabook.org/en/500L/a-python-interpreter-written-in-python.html)
