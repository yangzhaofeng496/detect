# detect

## 作用

ROS1 图像检测节点，按配置加载 YOLO、OCR、DINO 等检测器，并通过统一服务返回检测结果。

## 运行

当前包没有独立 launch 文件。在已加载 ROS1 和工作空间环境的终端中运行：

```bash
rosrun detect detect.py
```

配置文件：

- `config/detect.yaml`：节点、话题和检测器列表。
- `config/yolo_detector.yaml`：YOLO 参数。
- `config/ocr_detector.yaml`：OCR 模型参数。
- `config/dino_detector.yaml`：DINO 模型和图片库。
- `config/thickness_detector.yaml`：测厚参数，默认未在主配置中启用。

## 接口

订阅：

- `/camera/image_raw` (`sensor_msgs/Image`)：彩色图像。
- `/camera/depth/image_raw` (`sensor_msgs/Image`)：深度图像。
- `/detect/context` (`std_msgs/String`)：检测上下文。

服务：

- `/detectPic` (`moma_service/DetectPic`)：执行指定类型的图像检测并返回目标列表。

离线模式可使用 `--offline`、`--color`、`--depth`、`--detect-type`、`--save-prefix` 等参数。
