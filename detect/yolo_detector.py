#! /usr/bin/env python
import csv
import json
import os
import threading
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import cv2
import yaml
import torch
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont

try:
    from . import yolo_sub_detector
    from .path_utils import load_yaml_config, resolve_path
except ImportError:
    import yolo_sub_detector
    from path_utils import load_yaml_config, resolve_path

class YoloDetector:
    def __init__(self, config_path):
        print("正在初始化YOLO检测器...")
        self.config = self._load_config(config_path)
        self.weights = self.config['weights']
        self.imgsz = self.config['imgsz']
        self.conf = self.config['confidence']
        self.iou = self.config['iou']
        self.device = self.config['device'] or self._auto_select_device()
        self.font_size = self.config['font_size']
        self.output_dir = self.config['output_dir']

        # 从 config 的 cls_map 构建索引映射
        # cls_map 示例： {0: {code: "bolt", name: "螺栓"}, 1: {...}}
        self.cls_map = self.config.get('cls_map', {})

        # 初始化模型
        self.model = self._load_model()
        
        # 可视化资源
        self.font, self.has_zh_font = self._load_font()
        self.palette = self._get_palette()
        
        # 线程锁（用于多线程环境）
        self.lock = threading.Lock()
        
        print(f"YOLO检测器初始化完成: ")
        print(f"  - 模型: {self.weights}")
        print(f"  - 推理图像尺寸: {self.imgsz}")
        print(f"  - 置信度阈值: {self.conf}")
        print(f"  - 设备: {self.device}")
        print(f"  - 中文字体: {'可用' if self.has_zh_font else '不可用(使用英文)'}")
    
    def _load_config(self, config_path):
        """加载配置文件（支持多种编码）"""
        return load_yaml_config(config_path)

    def _auto_select_device(self) -> str:
        """自动选择推理设备"""
        return "cuda" if torch.cuda.is_available() else "cpu"
    
    def _load_model(self) -> YOLO:
        """加载YOLO模型"""
        try:
            model = YOLO(self.weights)
            model.to(self.device)
            return model
        except Exception as e:
            raise RuntimeError(f"加载模型失败: {e}")
    
    def _load_font(self) -> Tuple[Optional[ImageFont.FreeTypeFont], bool]:
        """加载字体"""
        candidate_paths = [
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/arphic/ukai.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
        for p in candidate_paths:
            if os.path.exists(p):
                try:
                    return ImageFont.truetype(p, size=self.font_size), "CJK" in p
                except Exception:
                    pass
        try:
            return ImageFont.load_default(), False
        except Exception:
            return None, False
    
    def _get_palette(self) -> List[Tuple[int, int, int]]:
        """生成调色板"""
        preset = [
            (56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255),
            (49, 210, 207), (10, 249, 72), (23, 204, 146), (186, 55, 2),
            (52, 31, 151), (155, 155, 155), (87, 175, 74), (23, 23, 23),
        ]
        return preset

    # ---------------- subtask helpers ----------------
    def _perform_sub_task(self, sub_task: str, crop_img: np.ndarray, save_path_prefix: str) -> Dict[str, Any]:
        """Dispatch sub-task handling to specific function based on `task` name.
        Returns a tuple: ({"status": str, "confidence": float, ...}, "ok")
        """
        # try:
        if sub_task == "anti_loosen_mark":
            return yolo_sub_detector._check_anti_loosen_mark(crop_img, save_path_prefix)
            # if sub_task == "wire_integrity":
            #     return yolo_sub_detector._check_wire_integrity(crop_img)
            # if sub_task == "cotter_pin_angle":
            #     return yolo_sub_detector._check_cotter_pin_angle(crop_img)
        else:
            return "正常", "ok"
        # except Exception as e:
        #     # print('yolo_detector存在异常:' + str(e)) 
        #     return "异常", "error"

    def _draw_chinese_label(
        self,
        img_bgr: np.ndarray,
        xyxy: Tuple[float, float, float, float],
        text: str,
        color: Tuple[int, int, int],
        box_alpha: float = 0.25,
    ):
        """绘制标签"""
        x1, y1, x2, y2 = map(int, xyxy)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img_bgr.shape[1]-1, x2), min(img_bgr.shape[0]-1, y2)

        # 半透明框
        overlay = img_bgr.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness=-1)
        cv2.addWeighted(overlay, box_alpha, img_bgr, 1 - box_alpha, 0, img_bgr)

        # 框边线
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, thickness=2)

        # 绘制文本
        if self.font:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            draw = ImageDraw.Draw(pil_img)

            # 文本背景
            text_bbox = draw.textbbox((0, 0), text, font=self.font)
            text_w = text_bbox[2] - text_bbox[0]
            text_h = text_bbox[3] - text_bbox[1]
            pad = 4
            bg_x1, bg_y1 = x1, max(0, y1 - text_h - 2 * pad)
            bg_x2, bg_y2 = x1 + text_w + 2 * pad, bg_y1 + text_h + 2 * pad
            draw.rectangle([bg_x1, bg_y1, bg_x2, bg_y2], fill=(0, 0, 0, 180))
            draw.text((bg_x1 + pad, bg_y1 + pad), text, font=self.font, fill=(255, 255, 255))

            img_bgr[:, :, :] = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        else:
            # 降级使用OpenCV绘制文本
            cv2.putText(img_bgr, text, (x1 + 5, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    @staticmethod
    def ensure_dir(d: str):
        """确保目录存在"""
        Path(d).mkdir(parents=True, exist_ok=True)
    
    @staticmethod
    def is_image(p: str) -> bool:
        """判断是否为图片文件"""
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
        return Path(p).suffix.lower() in exts
    
    @staticmethod
    def collect_images(source: str) -> List[str]:
        """收集图片文件"""
        p = Path(source)
        if p.is_file():
            return [str(p)] if YoloDetector.is_image(str(p)) else []
        if p.is_dir():
            files = [str(f) for f in p.glob("*.png")]

            def _num_key(fp: str):
                stem = Path(fp).stem
                m = re.search(r"(\d+)", stem)
                return int(m.group(1)) if m else float("inf")

            files.sort(key=_num_key)
            return files
        else:
            return []
        
    
    def infer(self, save_path_prefix: str, color_img: np.ndarray, depth_img: np.ndarray) -> Dict[str, Any]:
        try:
            # basic validation
            if color_img is None or not isinstance(color_img, np.ndarray):
                return {"analyse_result": "error", "message": f"推理输入图像异常", "detections": []}

            # ensure output directory exists
            save_dir, _ = save_path_prefix.rsplit('/', 1)
            os.makedirs(save_dir, exist_ok=True)

            # perform inference under lock
            with self.lock:
                results = self.model.predict(
                    source=color_img,
                    imgsz=self.imgsz,
                    conf=self.conf,
                    iou=self.iou,
                    verbose=False,
                    device=self.device,
                )

            if not results:
                return {"analyse_result": "error", "message": "推理未返回结果", "detections": []}

            res = results[0]
            detections: List[Dict[str, Any]] = []
            result_img = color_img.copy()

            analyse_result = "ok"
            message = ""

            boxes = getattr(res, "boxes", None)
            if boxes is None or len(boxes) == 0:
                return {"analyse_result": "error", "message": "推理返回结果数量为0", "detections": []}
            
            # extract arrays once
            clss = boxes.cls.cpu().numpy().astype(int)
            confs = boxes.conf.cpu().numpy()
            xyxy = boxes.xyxy.cpu().numpy()

            for i in range(len(clss)):
                class_id = clss[i]
                cf = float(confs[i])
                x1f, y1f, x2f, y2f = xyxy[i].tolist()

                # normalize and clamp integer bbox values
                x1 = max(0, int(round(x1f)))
                y1 = max(0, int(round(y1f)))
                w = max(1, int(round(x2f - x1f)))
                h = max(1, int(round(y2f - y1f)))

                # lookup class mapping; skip unknown classes but continue
                cls_entry = self.cls_map.get(class_id)
                if not cls_entry:
                    return {"analyse_result": "error", "message": f"推理返回类别为{class_id}，该类别未定义", "detections": []}

                zh_name = cls_entry.get("name", f"类：{class_id}")
                en_code = cls_entry.get("code", f"class:{class_id}")
                name = zh_name if self.has_zh_font else en_code
                label_text = f"{name} {cf:.2f}"

                status = ""
                sub_task = cls_entry.get("sub_task", "")
                if sub_task:
                    crop = color_img[y1:y1 + h, x1:x1 + w]
                    try:
                        status, sub_result = self._perform_sub_task(sub_task, crop, save_path_prefix + f"_{i}")
                    except Exception as e:
                        return {"analyse_result": "error", "message": f"推理子任务异常: {str(e)}", "detections": []}
                    if sub_result == "error":
                        analyse_result = "error"
                        
                is_abnormal = bool(status and status != "正常")
                if is_abnormal:
                    analyse_result = "error"

                det = {
                    "abnormal_obj": is_abnormal,
                    "type_code": en_code,
                    "type_name": zh_name,
                    "confidence": cf,
                    "bbox": [x1, y1, w, h],
                    "status": status,
                }
                detections.append(det)
               

                # draw box using integer coords
                color = self.palette[class_id % len(self.palette)]
                self._draw_chinese_label(result_img, (x1, y1, x1 + w, y1 + h), label_text, color)
            # 统计各状态并构建 message，空 status 显示为 "正常"
            status_counts: Dict[str, int] = {}
            for d in detections:
                s = d.get("status") or ""
                key = s if s else "正常"
                status_counts[key] = status_counts.get(key, 0) + 1

            parts = [f"共{len(detections)}个目标"]
            for k, v in status_counts.items():
                parts.append(f"{v}个{k}")
            message = '，'.join(parts)
            
            # save visualization
            out_path = save_path_prefix + "_result.png"
            cv2.imwrite(out_path, result_img)
            point_id = os.path.basename(save_path_prefix)
            print(f"第{point_id}个项点处理完成: 目标总数{len(detections)}, {message}")

            return {"analyse_result": analyse_result, "message": message, "detections": detections}
        except Exception as e:
            return {"analyse_result": "error", "message": f"推理过程发生其他异常: {e}", "detections": []}

    def run_batch_inference(
        self,
        source: str,
        out_dir: str,
        save_csv: bool = True,
        save_json: bool = True
    ):
        """ 批量推理（保持原有功能）"""
        # self.ensure_dir(out_dir)
        # # set output directory for saved visualizations
        # self.output_dir = out_dir
        
        # 收集图片
        img_list = self.collect_images(source)    
        if len(img_list) == 0:
            print(f"警告：未找到图片: {source}")
            return
        
        # 准备输出文件
        csv_path = os.path.join(out_dir, "detections.csv")
        json_path = os.path.join(out_dir, "detections.json")
        
        csv_file = open(csv_path, "w", newline="", encoding="utf-8") if save_csv else None
        csv_writer = csv.writer(csv_file) if csv_file else None
        if csv_writer:
            csv_writer.writerow(["image_path", "type_code", "type_name",
             "confidence", "x1", "y1", "w", "h", "status"])
        
        jsonl_f = open(json_path, "w", encoding="utf-8") if save_json else None
        
        cnt = 0
        # 处理每张图片
        for im_path in img_list:
            img_bgr = cv2.imread(im_path)
            if img_bgr is None:
                print(f"警告：读取失败: {im_path}")
                continue
            cnt += 1
            depth_img = None
            print(f"检测第{cnt}张图像")
            result = self.infer(source + f'/result/{cnt}', img_bgr, depth_img)
            # 保存结果
            if csv_writer:
                for det in result['detections']:
                    csv_writer.writerow([
                        im_path,
                        det["type_code"],
                        det["type_name"],
                        f"{det['confidence']:.4f}",
                        int(det["bbox"][0]),
                        int(det["bbox"][1]),
                        int(det["bbox"][2]),
                        int(det["bbox"][3]),
                        det.get("status", "")
                    ])
            
            if jsonl_f:
                json.dump({
                    "image_path": im_path,
                    "detections": result['detections']
                }, jsonl_f, ensure_ascii=False)
                jsonl_f.write("\n")
        
        # 关闭文件
        if csv_file:
            csv_file.close()
        if jsonl_f:
            jsonl_f.close()
        
        print(f"\n批量处理完成！结果保存在: {out_dir}")


if __name__ == "__main__":
    # 创建检测器实例
    config_path = resolve_path('$(find detect)/config/yolo_detector.yaml')
    detector = YoloDetector(config_path)
    
    # 批量测试
    detector.run_batch_inference(
        source="/home/shansu/moma_ws/records/HXD3C/model_comparison",
        out_dir="/home/shansu/moma_ws/records/HXD3C/model_comparison/result_new2"
    )
