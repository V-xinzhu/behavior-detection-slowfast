
# 🧠 SlowFast Behavior detection System

这是一个基于 **SlowFast 网络** 的行为识别系统，集成了 **YOLOv11 实时检测** 和 **DeepSORT 目标跟踪** 技术，用于视频中人物行为的实时识别与追踪。

------

## 📌 项目结构概览

本项目主要包含以下模块：

### 🧱 核心模型

- `slowfast/`：SlowFast 视频行为识别网络代码(从pyslowfast项目复制了推理的部分)。
- `deep_sort/`：DeepSORT 目标跟踪算法实现（YOLOTrack不需要它）。
- `ultralytics/YOLO`：YOLOv11 检测器（通过 Ultralytics 接口加载）。

### 🛠 工具模块

- [Tools/InferenceManager.py](javascript:void(0))：
  初始化大类，加载 YOLO、SlowFast 和 DeepSORT模型参数，以及视频加载和保存等功能的初始化。
- [Tools/visulize.py](javascript:void(0))：
  可视化工具，绘制边界框和动作标签。
- [Tools/readYaml.py](javascript:void(0))：
  读取配置文件。
- `utils/`：
  通用工具函数，包括图像加载(camera.py)
  输出数据json处理(Data_process.py)、
  日志记录(logger.py)、
  数据格式转换等(Format_conversion.py)。

### 📊 其他文件

- `demo`：存放行为检测的输入视频源和输出视频数据
- `label`：存放行为检测的识别标签
- `weight`：存放模型参数文件
- `RUN`：程序接口运行文件夹包含 method1,method2,YOLOTrack

### 🖼 示例截图

​		这里的BGR通道是反的.

- ![image-20250715113659641](C:\Users\xinzhu\AppData\Roaming\Typora\typora-user-images\image-20250715113659641.png)

------

## 🧰 主要功能

1. ✅ method1：实时监测,收到get请求后发送当前识别结果的json格式数据

- 使用 **YOLOv11** 进行目标检测；

- 使用 **DeepSORT** 跟踪目标；

- 使用 **SlowFast** 对目标区域进行行为识别。

- 📺 支持本地摄像头、视频文件或实时流媒体输入。

- 📈 可视化输出：

  ​	显示目标 ID、类别及行为预测；

  ​	支持输出为 Base64 编码图像并以 HTTP 接口提供结果。

1. ✅method2:收到get请求再发送但前识别结果
2. ✅YOLOTrack:收到上游服务的get请求从**redis**获取数据再发送但前识别结果.

- 使用 **YOLOTrack** 进行目标检测和目标跟踪；

------

## 🧪 运行方式

### 🔧 安装依赖

```
bash
conda env create -f slowfast.yml
```

### 🚀 启动服务（HTTP API）

```
bashcd RUN/method1/ or method2
python run.py
```

服务将启动一个 端口为6666 的 HTTP 服务器，监听 `/get_result` 接口，返回 JSON 格式的检测 + 行为识别结果。

------

## 📡 接口说明

### GET `/get_result`

**请求示例：**

```
bash

curl http://localhost:6666/get_result
```

**响应示例：**

```
json{
  "img": "<base64-encoded image>",
  "detections": [
    {
      "x1": 120,
      "y1": 50,
      "x2": 200,
      "y2": 300,
      "class_id": 0,
      "track_id": 1,
      "ava_label": "stand"
    }
  ]
}
```

------

## 📝 配置说明

详见 [config.yaml](javascript:void(0)) 文件。常用配置项如下：

```
yamldevice: "cuda" # 或 cpu
input_path: 0 # 摄像头编号或视频路径

yolo:
  model_type: "yolov11" #就是个输出提示没有具体作用
  model_path: "weights/yolov11s.pt" #模型路径

slowfast:
  model_cfg: "configs/Kinetics/SLOWFAST_4x16_R50.yaml" #slowfast模型配置文件 如果要使用自训练模型需要参考https://github.com/facebookresearch/SlowFast 安装对应的库文件，包括 detectron2和pytorchvideo
  model_path: "checkpoints/slowfast_r50.pth" #也可以使用pyslowfast的初始模型

deepsort:
  REID_CKPT: "deep_sort/deep/checkpoint/ckpt.t7" #deepsort 模型配置路径
```

------

## 📂 文件结构简要说明

| 文件夹       | 描述                                        |
| :----------- | :------------------------------------------ |
| `slowfast/`  | SlowFast 主体网络、损失函数、数据集构建器等 |
| `deep_sort/` | DeepSORT 跟踪器及特征提取模块               |
| `Tools/`     | 推理控制类、可视化方法                      |
| `utils/`     | 图像处理、数据封装、日志记录等通用工具      |
| `RUN/`       | 启动脚本和服务端程序                        |

------



------

## 👥 作者

- **xinzhu**
- GitHub: [@xinzhu](https://github.com/xinzhu)

------

## 📬 联系方式

如有任何问题，欢迎提交 Issue 或联系邮箱（2335117742@qq.com）。
