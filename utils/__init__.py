# 从当前目录导入registry模块
from . import registry

# 声明对外暴露的模块，这样其他文件可以直接通过from utils import registry导入
__all__ = ['registry']
