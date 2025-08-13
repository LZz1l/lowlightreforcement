from typing import Dict, Callable, Optional, Iterator, Tuple, Any


class Registry:
    """通用组件注册表，用于注册和管理各类组件（如模型、损失函数、数据集等）。

    支持两种注册方式：
    1. 装饰器模式：@registry.register(name="xxx")
    2. 直接调用：registry.register(obj, name="xxx")

    Args:
        name (str): 注册表名称（用于标识注册表用途，如"model"、"loss"）
    """

    def __init__(self, name: str):
        self._name = name
        self._registry: Dict[str, Callable] = {}  # 存储注册的组件：{组件名: 组件类/函数}

    @property
    def name(self) -> str:
        """返回注册表名称"""
        return self._name

    def register(
        self, 
        obj: Optional[Callable] = None, 
        name: Optional[str] = None, 
        allow_override: bool = False
    ) -> Callable:
        """注册组件（支持装饰器和直接调用两种方式）

        Args:
            obj (Callable, optional): 待注册的组件（类或函数），默认为None（用于装饰器模式）
            name (str, optional): 组件注册名称，默认为None（使用obj.__name__作为名称）
            allow_override (bool, optional): 是否允许覆盖已注册的同名组件，默认为False

        Returns:
            Callable: 注册后的组件（保持原功能不变）
        """
        if obj is None:
            # 装饰器模式：@registry.register(...)
            def wrapper(wrapped_obj: Callable) -> Callable:
                self._register(
                    obj=wrapped_obj,
                    name=name or wrapped_obj.__name__,
                    allow_override=allow_override
                )
                return wrapped_obj
            return wrapper
        else:
            # 直接调用模式：registry.register(obj, ...)
            self._register(
                obj=obj,
                name=name or obj.__name__,
                allow_override=allow_override
            )
            return obj

    def _register(self, obj: Callable, name: str, allow_override: bool) -> None:
        """实际执行注册的内部方法（不对外暴露）"""
        if name in self._registry and not allow_override:
            raise ValueError(
                f"组件 '{name}' 已在注册表 '{self.name}' 中注册，"
                f"若需覆盖请设置 allow_override=True"
            )
        self._registry[name] = obj

    def get(self, name: str) -> Callable:
        """根据名称获取注册的组件

        Args:
            name (str): 组件注册名称

        Returns:
            Callable: 注册的组件（类或函数）

        Raises:
            KeyError: 若名称未在注册表中注册
        """
        if name not in self._registry:
            raise KeyError(
                f"组件 '{name}' 未在注册表 '{self.name}' 中注册，"
                f"可用组件: {list(self._registry.keys())}"
            )
        return self._registry[name]

    def keys(self) -> Iterator[str]:
        """返回所有注册组件的名称迭代器"""
        return iter(self._registry.keys())

    def values(self) -> Iterator[Callable]:
        """返回所有注册组件的迭代器"""
        return iter(self._registry.values())

    def __contains__(self, name: str) -> bool:
        """检查组件名称是否已注册"""
        return name in self._registry

    def __iter__(self) -> Iterator[Tuple[str, Callable]]:
        """迭代所有注册的组件（返回(name, obj)元组）"""
        return iter(self._registry.items())

    def __len__(self) -> int:
        """返回注册组件的数量"""
        return len(self._registry)

    def __repr__(self) -> str:
        """返回注册表的字符串表示（便于调试）"""
        return f"Registry(name='{self.name}', size={len(self)}, components={list(self.keys())})"


# 实例化常用注册表（统一管理项目中的核心组件）
MODEL_REGISTRY = Registry('model')  # 用于注册模型类（如LAENet、Retinexformer）
LOSS_REGISTRY = Registry('loss')   # 用于注册损失函数类（如RetinexPerturbationLoss）
DATASET_REGISTRY = Registry('dataset')  # 用于注册数据集类（如LOLv2Dataset）
ARCH_REGISTRY = Registry('arch')