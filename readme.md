# 如何部署神经网络程序

通常用PyTorch、Tensorflow开发神经网络，得到模型。

如果是网络服务，可直接用python开发服务程序。

如需开发客户端程序，Windows上首选基于OpenCV的部署方案，Windows ML 也值得尝试。

## OpenCV方案

OpenCV的DNN模块直接支持 Caffe、Darknet、Onnx、Tensorflow、Torch 模型。

* 其中Onnx作为中立的交换模型，可用[工具](https://github.com/onnx/onnxmltools)由其他模型转换。

## Windows ML方案(UWP)

集成到Windows系统的高性能API，在 Windows 应用中实现机器学习。

PyTorch Image Classification演示 训练PyTorch模型、转换为Onnx模型、应用部署 的完整流程。

## 注意事项

集成时，主要关注输入和输出层。[Netron](https://github.com/lutzroeder/netron)是不错的模型查看工具。

## 参考链接

* [opencvsharp_samples-CaffeSample, FaceDetectionDNN](https://github.com/shimat/opencvsharp_samples)
* [Windows ML简介](https://learn.microsoft.com/zh-cn/windows/ai/windows-ml/)
* [WinML-PyTorch Image Classification](https://github.com/microsoft/Windows-Machine-Learning/tree/master/Samples/Tutorial%20Samples/PyTorch%20Image%20Classification)
* [ONNXMLTools](https://github.com/onnx/onnxmltools)
* []()

