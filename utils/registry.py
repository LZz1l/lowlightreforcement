class Registry:
    """自定义注册器类，用于注册网络架构"""

    def __init__(self, name):
        self._name = name
        self._module_dict = dict()

    def __len__(self):
        return len(self._module_dict)

    def __contains__(self, key):
        return self.get(key) is not None

    def __repr__(self):
        format_str = self.__class__.__name__ + f'(name={self._name}, ' \
                                               f'items={list(self._module_dict.keys())})'
        return format_str

    @property
    def name(self):
        return self._name

    @property
    def module_dict(self):
        return self._module_dict

    def get(self, key):
        """根据key获取注册的模块"""
        return self._module_dict.get(key, None)

    def register_module(self, name=None, force=False):
        """注册模块的装饰器"""

        def _register_module(module):
            if not isinstance(module, type):
                raise TypeError(f'module must be a class, but got {type(module)}')

            module_name = name if name is not None else module.__name__
            if module_name in self._module_dict and not force:
                raise KeyError(f'{module_name} is already registered in {self.name}')
            self._module_dict[module_name] = module
            return module

        return _register_module


# 创建网络架构注册器（替代basicsr的ARCH_REGISTRY）
ARCH_REGISTRY = Registry('arch')
