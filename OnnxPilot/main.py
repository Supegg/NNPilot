import onnx
import onnxruntime

# 使用异常处理的方法进行检验
try:
    # 当模型不可用时，将会报出异常
    onnx.checker.check_model("super_resolution.onnx")
except onnx.checker.ValidationError as e:
    print("The model is invalid: %s"%e)
else:
    print("The model is valid!")


ort_session = onnxruntime.InferenceSession("super_resolution.onnx")

# 将张量转化为ndarray格式
def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# 构建输入的字典和计算输出结果
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
ort_outs = ort_session.run(None, ort_inputs)

# 比较使用PyTorch和ONNX Runtime得出的精度
np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)

print("Exported model has been tested with ONNXRuntime, and the result looks good!")    