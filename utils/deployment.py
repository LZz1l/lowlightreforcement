import torch
import onnx
import os
from onnxruntime import InferenceSession
from typing import Optional, Tuple, Union
import numpy as np


def convert_to_onnx(
    model: torch.nn.Module,
    input_shape: Tuple[int, int, int, int] = (1, 3, 256, 256),
    save_path: str = 'laenet.onnx',
    dynamic_batch: bool = True
) -> str:
    """
    将PyTorch模型转为ONNX格式（适配低光增强模型）

    Args:
        model: 训练好的PyTorch模型（如LAENet、Retinexformer）
        input_shape: 输入张量形状 [B, C, H, W]
        save_path: ONNX模型保存路径
        dynamic_batch: 是否支持动态批次大小

    Returns:
        保存的ONNX模型路径
    """
    model.eval()
    dummy_input = torch.randn(*input_shape)

    # 配置动态维度（支持任意H、W，可选动态B）
    dynamic_axes = {
        'input': {2: 'height', 3: 'width'},
        'output': {2: 'height', 3: 'width'}
    }
    if dynamic_batch:
        dynamic_axes['input'][0] = 'batch_size'
        dynamic_axes['output'][0] = 'batch_size'

    # 导出ONNX
    torch.onnx.export(
        model,
        dummy_input,
        save_path,
        opset_version=12,  # 兼容更多部署框架
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=dynamic_axes
    )

    # 验证ONNX模型有效性
    onnx_model = onnx.load(save_path)
    onnx.checker.check_model(onnx_model)
    print(f'ONNX模型已保存至: {save_path}')
    return save_path


def enable_mkldnn_acceleration(model: torch.nn.Module) -> torch.nn.Module:
    """启用MKLDNN加速（CPU推理优化，适配低光图像快速处理）"""
    if not torch.backends.mkldnn.enabled:
        torch.backends.mkldnn.enabled = True
        print("MKLDNN加速已启用（CPU推理）")
    model = model.to('cpu').eval()
    return model


def npu_inference(
    onnx_path: str,
    input_tensor: Union[torch.Tensor, np.ndarray],
    device_id: int = 0
) -> torch.Tensor:
    """
    NPU推理（适配华为Ascend等设备，需安装Ascend Toolkit）

    Args:
        onnx_path: ONNX模型路径
        input_tensor: 输入张量（PyTorch或NumPy格式）
        device_id: NPU设备ID

    Returns:
        推理结果（PyTorch张量）
    """
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX模型不存在: {onnx_path}")

    # 转换输入为NumPy格式
    if isinstance(input_tensor, torch.Tensor):
        input_tensor = input_tensor.cpu().numpy()

    # 配置ONNX Runtime使用NPU
    try:
        session = InferenceSession(
            onnx_path,
            providers=['AscendExecutionProvider'],
            provider_options=[{'device_id': device_id}]
        )
    except Exception as e:
        raise RuntimeError(f"NPU推理初始化失败: {str(e)}")

    # 执行推理
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: input_tensor})
    return torch.from_numpy(output[0])


def quantize_model(
    model: torch.nn.Module,
    save_path: str = 'laenet_quantized.pth',
    qconfig_spec: Optional[dict] = None
) -> torch.nn.Module:
    """
    动态量化模型（INT8），加速CPU推理并减小体积（适配低光增强模型的卷积层）

    Args:
        model: 待量化的PyTorch模型
        save_path: 量化模型保存路径
        qconfig_spec: 自定义量化配置（默认量化Conv2d和Linear层）

    Returns:
        量化后的模型
    """
    from torch.quantization import quantize_dynamic, QConfig
    import torch.nn as nn

    model.eval()
    # 默认量化配置（适配项目中的卷积层和全连接层）
    if qconfig_spec is None:
        qconfig = QConfig(
            activation=nn.quantized.FloatFunctional.with_args(dtype=torch.quint8),
            weight=nn.quantized.default_weight_observer.with_args(dtype=torch.qint8)
        )
        qconfig_spec = {nn.Conv2d: qconfig, nn.Linear: qconfig}

    # 动态量化
    quantized_model = quantize_dynamic(
        model,
        qconfig_spec=qconfig_spec,
        dtype=torch.qint8
    )

    # 保存量化模型
    torch.save(quantized_model.state_dict(), save_path)
    print(f'量化模型已保存至: {save_path}')
    return quantized_model