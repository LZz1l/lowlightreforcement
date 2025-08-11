class Registry:
    """通用组件注册表，用于注册和获取类/函数（如模型、损失函数）"""
    def __init__(self, name):
        self._name = name
        self._registry = {}  # 存储注册的组件：{组件名: 组件类/函数}

    @property
    def name(self):
        return self._name

    def register(self, obj=None, name=None):
        """注册组件（支持装饰器和直接调用两种方式）"""
        if obj is None:
            # 作为装饰器使用：@registry.register(name='xxx')
            def wrapper(obj):
                self._register(obj, name or obj.__name__)
                return obj
            return wrapper
        else:
            # 直接调用：registry.register(obj, name='xxx')
            self._register(obj, name or obj.__name__)
            return obj

    def _register(self, obj, name):
        if name in self._registry:
            raise ValueError(f"组件 {name} 已在 {self.name} 中注册")
        self._registry[name] = obj

    def get(self, name):
        """根据名称获取注册的组件"""
        if name not in self._registry:
            raise KeyError(f"组件 {name} 未在 {self.name} 中注册，可用组件: {list(self._registry.keys())}")
        return self._registry[name]

    def __contains__(self, name):
        return name in self._registry

    def __iter__(self):
        return iter(self._registry.items())


# 实例化常用注册表（适配项目中的模型、损失函数等）
MODEL_REGISTRY = Registry('model')
LOSS_REGISTRY = Registry('loss')
DATASET_REGISTRY = Registry('dataset')