import torch
import onnx
from onnxruntime import InferenceSession

def convert_to_onnx(model, input_shape=(1, 3, 256, 256), save_path='laenet.onnx'):
    """将PyTorch模型转为ONNX格式，便于部署"""
    model.eval()
    dummy_input = torch.randn(*input_shape)
    # 导出ONNX
    torch.onnx.export(
        model,
        dummy_input,
        save_path,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {2: 'height', 3: 'width'}, 'output': {2: 'height', 3: 'width'}}
    )
    # 验证ONNX模型
    onnx_model = onnx.load(save_path)
    onnx.checker.check_model(onnx_model)
    print(f'ONNX模型已保存至: {save_path}')
    return save_path

def enable_mkldnn_acceleration(model):
    """启用MKLDNN加速（CPU推理优化）"""
    if not torch.backends.mkldnn.enabled:
        torch.backends.mkldnn.enabled = True
        print("MKLDNN加速已启用")
    model = model.to('cpu')
    model.eval()
    return model

def npu_inference(onnx_path, input_tensor):
    """NPU推理（以华为Ascend为例，需安装Ascend Toolkit）"""
    # 配置ONNX Runtime使用NPU
    session = InferenceSession(
        onnx_path,
        providers=['AscendExecutionProvider'],
        provider_options=[{'device_id': 0}]  # 指定NPU设备ID
    )
    # 推理
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: input_tensor.numpy()})
    return torch.from_numpy(output[0])


from torch.quantization import quantize_dynamic, QConfig
import torch.nn as nn

def quantize_model(model, save_path='laenet_quantized.pth'):
    """动态量化模型（INT8），加速CPU推理并减小体积"""
    model.eval()
    # 配置量化参数
    qconfig = QConfig(
        activation=nn.quantized.FloatFunctional.with_args(dtype=torch.quint8),
        weight=nn.quantized.default_weight_observer.with_args(dtype=torch.qint8)
    )
    # 动态量化（仅量化线性层和LSTM等）
    quantized_model = quantize_dynamic(
        model,
        {nn.Conv2d, nn.Linear},  # 指定需要量化的层
        dtype=torch.qint8
    )
    # 保存量化模型
    torch.save(quantized_model.state_dict(), save_path)
    print(f'量化模型已保存至: {save_path}')
    return quantized_model