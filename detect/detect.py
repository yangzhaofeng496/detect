#! /usr/bin/env python
# ROS 1 (noetic) 迁移版 detect.py
import argparse
import importlib
import inspect
import os
import rospy
import yaml
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
from moma_service.srv import DetectPic, DetectPicResponse
from moma_service.msg import DetectObj
import cv2
import numpy as np
import sys
from pathlib import Path

try:
    from .path_utils import load_yaml_config, resolve_ros_substitutions
except ImportError:
    from path_utils import load_yaml_config, resolve_ros_substitutions

# Ensure workspace src/ is on Python path so package imports such as detect.* work
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


class DetectNode(object):
    """动态配置的检测节点，支持加载多种检测器"""

    def __init__(self, use_ros=True, config_path=None):
        self.use_ros = use_ros
        if self.use_ros:
            rospy.init_node("detect_node", anonymous=False)
            # 初始化CvBridge
            self.bridge = CvBridge()
        else:
            self.bridge = CvBridge()

        # 参数
        if config_path is None:
            if self.use_ros:
                config_path = rospy.get_param(
                    "~config_path", "$(find detect)/config/detect.yaml"
                )
            else:
                config_path = "$(find detect)/config/detect.yaml"
        self.config = self.load_config(config_path)
        # 检测器
        self.detectors = {}
        self.load_detectors()
        # 图像缓存
        self.color_image = None
        self.depth_image = None
        self.detect_context = {}
        # 话题/服务
        self.color_image_topic = self.config.get(
            "hik_color_topic", "/camera/image_raw"
        )
        self.depth_image_topic = self.config.get(
            "depth_topic", "/camera/depth/image_raw"
        )
        self.detect_service_name = self.config.get("detect_service_name", "/detectPic")

        if self.use_ros:
            self.context_sub = rospy.Subscriber(
                '/detect/context', String, self.context_callback, queue_size=1
            )
            if self.config.get("use_ros_topic", True):
                self.color_sub = rospy.Subscriber(
                    self.color_image_topic, Image, self.color_callback, queue_size=1
                )
                self.depth_sub = rospy.Subscriber(
                    self.depth_image_topic, Image, self.depth_callback, queue_size=1
                )
                rospy.loginfo("检测节点彩色图像话题订阅已启动：%s", self.color_image_topic)
                rospy.loginfo("检测节点深度图像话题订阅已启动：%s", self.depth_image_topic)
            else:
                rospy.loginfo("已禁用 ROS 图像订阅，使用离线输入模式")

            # 服务
            self.service = rospy.Service(
                self.detect_service_name, DetectPic, self.detect_service_callback
            )
            rospy.loginfo(
                "检测节点初始化完成，支持的检测器类型：%s", list(self.detectors.keys())
            )
            rospy.loginfo("检测节点识别服务已启动：%s", self.detect_service_name)
        else:
            print("检测节点已启动离线模式，支持的检测器类型：%s" % list(self.detectors.keys()))
        # rospy.loginfo(f"Python interpreter: {sys.executable}")
        # rospy.loginfo(f"Python version: {sys.version}")

    def load_config(self, config_path):
        """加载YAML配置文件"""
        try:
            return load_yaml_config(config_path)
        except Exception as e:
            rospy.logerr("加载配置失败: %s", e)
            return {}

    @staticmethod
    def load_image(path, color=True):
        """离线加载图像（RGB/深度）"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"图像不存在: {path}")
        if color:
            img = cv2.imread(path, cv2.IMREAD_COLOR)
        else:
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise RuntimeError(f"图像读取失败: {path}")
        return img

    def run_offline(self, save_path_prefix, detect_type, color_path, depth_path=None):
        """离线运行检测器，直接从本地图像读取"""
        self.color_image = self.load_image(color_path, color=True)
        self.depth_image = None
        if depth_path:
            depth_raw = self.load_image(depth_path, color=False)
            depth_scale = self.config.get("depth_scale", 0.001)
            depth_m = depth_raw.astype(np.float32) * depth_scale
            if depth_raw.dtype == np.uint16:
                depth_m[depth_raw == 0] = np.nan
            self.depth_image = depth_m

        save_dir = os.path.dirname(save_path_prefix)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        req = type("Req", (), {})()
        req.detect_type = detect_type
        req.save_path_prefix = save_path_prefix
        result = self.detect_service_callback(req)
        print("离线检测返回结果:", result)
        return result

    def run_offline_batch(self, source_dir, output_dir, detect_type, depth_path=None):
        """批量处理 source_dir 下所有图片，并把结果按子目录写入 output_dir。"""
        source_dir = os.path.abspath(source_dir)
        output_dir = os.path.abspath(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        image_exts = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"}
        image_files = []
        for root, _, files in os.walk(source_dir):
            for name in files:
                if Path(name).suffix.lower() in image_exts:
                    image_files.append(os.path.join(root, name))

        if not image_files:
            raise FileNotFoundError(f"未在 {source_dir} 中找到图片文件")

        image_files.sort()
        for image_path in image_files:
            rel_path = os.path.relpath(image_path, source_dir)
            rel_dir = os.path.dirname(rel_path)
            rel_name = Path(rel_path).stem
            target_dir = os.path.join(output_dir, rel_dir if rel_dir != "." else "")
            if target_dir:
                os.makedirs(target_dir, exist_ok=True)
            save_prefix = os.path.join(target_dir, rel_name)
            self.run_offline(
                save_path_prefix=save_prefix,
                detect_type=detect_type,
                color_path=image_path,
                depth_path=depth_path,
            )

    def load_detectors(self):
        """加载检测器"""
        detectors_config = self.config.get("detectors", [])
        for detector_info in detectors_config:
            detect_type = detector_info.get("detect_type")
            class_path = detector_info.get("class_path")
            config_path = detector_info.get("config_path")
            if not all([detect_type, class_path, config_path]):
                rospy.logwarn("检测器配置不完整: %s", detector_info)
                continue
            try:
                # 动态导入检测器类
                module_name, class_name = class_path.rsplit(".", 1)
                module = importlib.import_module(module_name)
                detector_class = getattr(module, class_name)

                # 根据构造函数参数灵活初始化
                sig = inspect.signature(detector_class.__init__)
                params = [p.name for p in sig.parameters.values() if p.name != "self"]
                if detector_class.__name__ == "DINODetector":
                    detector_cfg = {}
                    if config_path and os.path.exists(config_path):
                        with open(config_path, "r", encoding="utf-8") as f:
                            detector_cfg = yaml.safe_load(f) or {}
                    detector = detector_class(
                        model_path=resolve_ros_substitutions(detector_cfg.get(
                            "model_path",
                            self.config.get(
                                "dino_model_path",
                                "$(find detect)/models/dino/facebook/dinov3-vits16",
                            ),
                        )),
                        image_dir=resolve_ros_substitutions(detector_cfg.get(
                            "image_dir",
                            self.config.get("dino_image_dir", "images/batch0"),
                        )),
                        feature_type=detector_cfg.get(
                            "feature_type",
                            self.config.get("dino_feature_type", "mean_patch"),
                        ),
                        config_path=config_path,
                    )
                elif "config_path" in params:
                    detector = detector_class(config_path)
                else:
                    raise TypeError("无法初始化检测器: %s" % class_path)

                self.detectors[detect_type] = detector
                rospy.loginfo("成功加载检测器: %s (%s)", detect_type, class_path)
            except Exception as e:
                rospy.logerr("加载检测器失败 %s: %s", detect_type, e)

    def color_callback(self, msg):
        """彩色图像回调，直接转为OpenCV格式"""
        try:
            self.color_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            rospy.logerr("转换彩色图像失败: %s", e)

    def depth_callback(self, msg):
        # TODO 理解深度图含义
        """深度图回调：passthrough -> numpy 原始类型 -> float(m) 并处理无效值"""
        try:
            depth_raw = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            # msg.encoding 也可以用来判断当前编码
            # 如果是 16UC1，depth_raw.dtype == np.uint16
            # 确认 depth_scale（驱动/相机）以决定是否除以 1000 或 use depth_scale param
            depth_scale = 0.001  # 若单位为 mm，除以1000换成 m；否则从驱动/参数读取
            depth_m = depth_raw.astype(np.float32) * depth_scale
            # 将 0 视为无效深度，设置为 NaN
            if depth_raw.dtype == np.uint16:
                depth_m[depth_raw == 0] = np.nan
            self.depth_image = depth_m
        except Exception as e:
            rospy.logerr("转换深度图失败: %s", e)

    def context_callback(self, msg):
        try:
            self.detect_context = dict(
                field.split('=', 1)
                for field in msg.data.split(';')
                if '=' in field
            )
            if not self.detect_context.get('train_type_name'):
                raise ValueError("上下文缺少车型")
            rospy.loginfo("已更新检测上下文: %s", self.detect_context)
        except (TypeError, ValueError) as e:
            rospy.logwarn("忽略无效检测上下文: %s", e)

    def detect_service_callback(self, req):
        """检测服务回调"""
        res = DetectPicResponse()
        # try:
        # 检查图像是否就绪
        if self.color_image is None:
            res.analyse_result = "error"
            res.message = "未接收到彩色图像"
            res.object_count = 0
            return res
        # 获取检测类型
        detect_type = req.detect_type.upper()
        save_path_prefix = req.save_path_prefix
        cv2.imwrite(save_path_prefix + "_color.png", self.color_image)
        if self.depth_image is not None:
            cv2.imwrite(save_path_prefix + "_depth.png", self.depth_image)
        # 检查检测器是否存在
        if detect_type not in self.detectors:
            res.analyse_result = "error"
            res.message = "不支持的检测类型: %s, 支持: %s" % (
                req.detect_type,
                list(self.detectors.keys()),
            )
            res.object_count = 0
            res.object_list = []
            return res
        # 获取检测器实例并执行推理
        detector = self.detectors[detect_type]
        save_result_prefix = os.path.join(
            os.path.dirname(save_path_prefix),
            "result",
            os.path.basename(save_path_prefix),
        )
        if detect_type in ("DINO", "OCR"):
            context = self.detect_context
            result = detector.infer(
                save_result_prefix,
                self.color_image,
                self.depth_image,
                train_type_name=context.get("train_type_name"),
                expected_point_name=context.get("point_name"),
            )
        else:
            result = detector.infer(
                save_result_prefix, self.color_image, self.depth_image
            )
        # 构建响应
        res.analyse_result = result.get("analyse_result", "error")
        res.message = result.get("message", "未检测到识别结果")
        detections = result.get("detections", [])
        res.object_count = len(detections)
        # 转换检测结果到 DetectObj[]
        res.object_list = []
        for det in detections:
            obj = DetectObj()
            # 支持 camelCase 或 snake_case 的输入字段
            obj.abnormal_obj = str(
                det.get("abnormalObj", det.get("abnormal_obj", False))
            ).lower()
            obj.type_code = str(det.get("typeCode", det.get("type_code", "")))
            obj.type_name = str(det.get("typeName", det.get("type_name", "")))
            obj.confidence = str(det.get("confidence", 0.0) * 100)
            # bbox 支持多种格式
            bbox = det.get("bbox", {})
            if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                x, y, w, h = bbox[:4]
            else:
                x = det.get("offsetX", det.get("offset_x", bbox.get("x", 0)))
                y = det.get("offsetY", det.get("offset_y", bbox.get("y", 0)))
                w = det.get("width", bbox.get("w", 0))
                h = det.get("height", bbox.get("h", 0))
            obj.offset_x = str(x)
            obj.offset_y = str(y)
            obj.width = str(w)
            obj.height = str(h)
            obj.status = str(det.get("status", ""))
            # thickness 字段处理
            if detect_type == "THICKNESS":
                obj.thickness = det.get("thickness", 0)
            else:
                obj.thickness = "0.0"
            res.object_list.append(obj)
        rospy.loginfo("point %s response: %s", os.path.basename(save_path_prefix), res)
        return res

        # # except Exception as e:
        #     self.get_logger().error(f"检测服务异常: {e}")
        #     response.analyse_result = "error"
        #     response.message = str(e)
        #     response.object_count = 0
        #     response.object_list = []
        #     return response


def parse_args():
    parser = argparse.ArgumentParser(description="模块化图像检测节点")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="启用离线图片检测模式，跳过 ROS 图像订阅",
    )
    parser.add_argument(
        "--color",
        help="离线推理单张RGB图像路径",
    )
    parser.add_argument(
        "--depth",
        help="离线推理单张深度图像路径",
    )
    parser.add_argument(
        "--detect-type",
        default="YOLO",
        help="检测类型，例如 YOLO、OCR、THICKNESS、DINO",
    )
    parser.add_argument(
        "--save-prefix",
        default="./offline_detect",
        help="离线检测结果保存前缀",
    )
    parser.add_argument(
        "--source-dir",
        default=None,
        help="离线批处理输入目录；若提供则会递归处理该目录下所有图片",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="离线批处理结果输出目录；结果会按原目录结构写入其中",
    )
    parser.add_argument(
        "--config-path",
        default="$(find detect)/config/detect.yaml",
        help="检测节点配置文件路径",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.offline:
        node = DetectNode(use_ros=False, config_path=args.config_path)
        if args.source_dir and args.output_dir:
            node.run_offline_batch(
                source_dir=args.source_dir,
                output_dir=args.output_dir,
                detect_type=args.detect_type,
                depth_path=args.depth,  # !!! 这里深度图也应该从source_dir中获取
            )
        else:
            if not args.color:
                raise ValueError("离线模式需要指定 --color 图像路径，或同时提供 --source-dir 与 --output-dir")
            node.run_offline(
                save_path_prefix=args.save_prefix,
                detect_type=args.detect_type,
                color_path=args.color,
                depth_path=args.depth,
            )
    else:
        node = DetectNode(use_ros=True, config_path=args.config_path)
        rospy.spin()


if __name__ == "__main__":
    main()
