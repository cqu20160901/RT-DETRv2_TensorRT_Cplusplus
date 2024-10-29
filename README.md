# RT-DETRv2_TensorRT_Cplusplus
RT-DETRv2 tensorrt C++ 部署


本示例中，包含完整的代码、模型、测试图片、测试结果。

TensorRT版本：TensorRT-8.6.1.6

# rt-detrv2 训练

训练参考官方开源代码。

# 导出onnx模型

在官方到处onnx 的基础上进行简单的调整，这里不需要动态batch，也不需要进行解码到输入分辨率，进行了如下调整：
![image](https://github.com/user-attachments/assets/e30f5f92-d1b7-48d3-856d-13071500807d)

```python

        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            # outputs = self.postprocessor(outputs, orig_target_sizes)
            outputs = torch.sigmoid(outputs['pred_logits']), outputs['pred_boxes']
            return outputs


    model = Model()
    data = torch.rand(1, 3, 640, 640)

    torch.onnx.export(
        model, 
        data,
        args.output_file,
        input_names=['images'],
        output_names=['output1', 'output2'],
        opset_version=16, 
        verbose=False,
        do_constant_folding=True,
    )

```

最终导出的onnx结构如下：

![image](https://github.com/user-attachments/assets/745c1746-f0b4-4f88-8ff7-f590f204f504)

# 测试 onnx 结果

onnx 测试脚本[【链接】](https://github.com/cqu20160901/RT-DETRv2_TensorRT_Cplusplus/tree/main/onnx_demo)

![image](https://github.com/user-attachments/assets/5baa105d-82ba-4af0-a5d3-b909b24f7a8f)

# TensorRT C++ 部署

tensorrt 环境部署参考官方文档，主要版本和cuda匹配。

## 1、修改Tensorrt使用版本

![image](https://github.com/user-attachments/assets/6c56e200-8914-4942-a642-931937c112ef)

## 2、修改代码中模型对应的路径

![image](https://github.com/user-attachments/assets/c144430c-9150-458b-ba85-f50a7ee4e9dc)

## 3、编译运行

```shellpower
# 编译
cd RT-DETRv2_TensorRT_Cplusplus
mkdir build
cd build
cmake ..
make


# 运行
./detr_trt
```

# tensorrt 运行结果

特别说明：本示例用fp16精度掉的非常多，默认使用的fp32

![image](https://github.com/user-attachments/assets/c0201497-bf09-4d7f-a8e7-de0e833b98a5)

# 运行时耗

本示例使用的是 rtdetrv2_r18vd_120e_coco.yml 模型，模型输入分别率640x640，显卡rtx4090，cuda12.5，图像预处理用cuda加速

![image](https://github.com/user-attachments/assets/f29015f0-7b3f-496c-beaf-963783e5f6d9)


