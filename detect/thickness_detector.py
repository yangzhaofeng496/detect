import cv2
import torch
import numpy as np
import math
import os
import threading
from typing import Dict, Any, List
from ultralytics import YOLO
import yaml

try:
    from .path_utils import load_yaml_config, resolve_path
except ImportError:
    from path_utils import load_yaml_config, resolve_path

# 导入 LingBot-Depth 模型 (请确保在 lingbot-depth 根目录下运行或环境变量已配置)
from mdm.model.v2 import MDMModel

# ==========================================
# 模块 1: 3D 几何测厚核心算子
# ==========================================
class AdvancedThicknessMeasurement:
    def __init__(self, depth_unit_scale=1000.0):
        self.intrinsics = None 
        self.depth_scale = depth_unit_scale

    def deproject_pixel_to_point(self, u, v, z_mm):
        x = (u - self.intrinsics['cx']) * z_mm / self.intrinsics['fx']
        y = (v - self.intrinsics['cy']) * z_mm / self.intrinsics['fy']
        return np.array([x, y, z_mm])

    def get_eroded_depth(self, depth_npy, start_x, start_y, direction_vec, inward_pixels=3):
        h, w = depth_npy.shape
        norm = math.sqrt(direction_vec[0]**2 + direction_vec[1]**2)
        if norm == 0: return 0
        dx, dy = direction_vec[0] / norm, direction_vec[1] / norm

        safe_x = max(0, min(w - 1, int(start_x + dx * inward_pixels)))
        safe_y = max(0, min(h - 1, int(start_y + dy * inward_pixels)))

        return depth_npy[safe_y, safe_x] * self.depth_scale

    def calculate_thickness(self, vis_img, depth_npy, mask_bin, num_samples=15):
        """
        计算单体目标的厚度。
        :return: (thickness值, minAreaRect的4个顶点坐标)
        """
        contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, None
        
        cnt = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(cnt)
        box_pts = np.int0(cv2.boxPoints(rect))

        p0, p1, p2, p3 = box_pts
        d01, d12 = np.linalg.norm(p0 - p1), np.linalg.norm(p1 - p2)

        if d01 < d12:
            short_len, long_vec = d01, p2 - p1
            start_base, end_base = (p0 + p1) / 2, (p3 + p2) / 2
        else:
            short_len, long_vec = d12, p1 - p0
            start_base, end_base = (p0 + p3) / 2, (p1 + p2) / 2
            
        total_vec = end_base - start_base
        ortho_vec = np.array([-total_vec[1], total_vec[0]])
        ortho_vec = ortho_vec / (np.linalg.norm(ortho_vec) + 1e-6)

        valid_thicknesses =[]

        for i in range(num_samples):
            ratio = 0.1 + 0.8 * (i / (num_samples - 1)) if num_samples > 1 else 0.5
            center_x, center_y = start_base[0] + total_vec[0] * ratio, start_base[1] + total_vec[1] * ratio
            
            p_start = np.array([center_x, center_y]) - ortho_vec * short_len * 0.8
            p_end = np.array([center_x, center_y]) + ortho_vec * short_len * 0.8
            
            num_line_points = int(short_len * 2)
            x_vals = np.linspace(p_start[0], p_end[0], num_line_points)
            y_vals = np.linspace(p_start[1], p_end[1], num_line_points)
            
            first_point, last_point, in_mask = None, None, False

            for k in range(num_line_points):
                cx, cy = int(x_vals[k]), int(y_vals[k])
                if 0 <= cx < mask_bin.shape[1] and 0 <= cy < mask_bin.shape[0]:
                    if mask_bin[cy, cx] > 0:
                        if not in_mask:
                            in_mask = True
                            first_point = (cx, cy)
                        last_point = (cx, cy)
            
            if first_point and last_point:
                scan_dir = p_end - p_start
                scan_dir = scan_dir / np.linalg.norm(scan_dir)
                
                # 向内腐蚀 1 个像素获取安全深度
                d1_mm = self.get_eroded_depth(depth_npy, first_point[0], first_point[1], scan_dir, 1)
                d2_mm = self.get_eroded_depth(depth_npy, last_point[0], last_point[1], -scan_dir, 1)

                if d1_mm > 0 and d2_mm > 0:
                    P1 = self.deproject_pixel_to_point(first_point[0], first_point[1], d1_mm)
                    P2 = self.deproject_pixel_to_point(last_point[0], last_point[1], d2_mm)
                    thickness = np.linalg.norm(P1 - P2)
                    valid_thicknesses.append(thickness)
                    
                    # 在总图上绘制采样线
                    cv2.line(vis_img, first_point, last_point, (0, 255, 0), 1)
                    cv2.circle(vis_img, first_point, 2, (0, 0, 255), -1)
                    cv2.circle(vis_img, last_point, 2, (0, 0, 255), -1)

        if valid_thicknesses:
            mean_val, std_val = np.mean(valid_thicknesses), np.std(valid_thicknesses)
            filtered =[v for v in valid_thicknesses if abs(v - mean_val) <= 2 * std_val + 1e-5]
            final_thickness = np.mean(filtered) if filtered else mean_val
            return final_thickness, box_pts
            
        return None, box_pts


# ==========================================
# 模块 2: ROS 节点推理类 (集成 YOLO + LingBot + 测厚)
# ==========================================
class ThicknessDetector:
    def __init__(self, config_path: str):
        print("正在初始化Thickness检测器...")
        self.lock = threading.Lock()
        
        # 1. 加载 YAML 配置
        self.config = self._load_config(config_path)
        
        # 2. 设置设备
        device_str = self.config.get('device', 'cuda')
        self.device = torch.device(device_str if torch.cuda.is_available() else 'cpu')
        
        # 3. 初始化内参 (从 YAML 读取)
        self.intr_dict = self.config['intrinsics']
        self.intrinsics_np = np.array([
            [self.intr_dict['fx'], 0.0, self.intr_dict['cx']],
            [0.0, self.intr_dict['fy'], self.intr_dict['cy']],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)

        # 4. 初始化 YOLO
        self.model = YOLO(self.config['yolo_model_path'])
        self.imgsz = self.config.get('imgsz', 640)
        self.conf = self.config.get('confidence', 0.25)
        self.iou = self.config.get('iou', 0.45)
        
        # 5. 类别映射与字体
        self.cls_map = self.config.get('cls_map', {})
        self.has_zh_font = False  # 进阶可以参考 YoloDetector 加入字体加载逻辑
        self.palette = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (0, 255, 255)]

        # 6. 初始化 LingBot 深度模型 (路径来自 YAML)
        self.depth_model = MDMModel.from_pretrained(self.config['lingbot_model_path']).to(self.device)
        self.depth_model.eval()

        # 7. 初始化 3D 测厚模块
        self.measurer = AdvancedThicknessMeasurement(depth_unit_scale=1000.0)
        self.measurer.intrinsics = self.intr_dict

    def _load_config(self, config_path):
        """增加鲁棒的 YAML 加载"""
        return load_yaml_config(config_path)
        
    def _draw_label(self, img, bbox, text, color):
        """ 绘制带背景的文字标签 """
        x1, y1, x2, y2 = bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, text, (x1, max(15, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    def infer(self, save_path_prefix: str, color_img: np.ndarray, depth_img: np.ndarray) -> Dict[str, Any]:
        try:
            # 1. 基础异常校验
            if color_img is None or not isinstance(color_img, np.ndarray) or depth_img is None:
                return {"analyse_result": "error", "message": "推理输入图像异常", "detections":[]}

            # 2. 保证输出目录存在
            save_dir, _ = save_path_prefix.rsplit('/', 1)
            os.makedirs(save_dir, exist_ok=True)

            # 3. 线程锁：执行 YOLO 推理
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
            boxes = getattr(res, "boxes", None)
            masks = getattr(res, "masks", None)
            
            if boxes is None or len(boxes) == 0 or masks is None:
                return {"analyse_result": "ok", "message": "正常：画面中未检测到目标", "detections":[]}

            # ---------------------------------------------------------
            # 4. 执行 AI 深度图补全 (仅整图运行一次，极大地节省 ROS 耗时)
            # ---------------------------------------------------------
            h, w = color_img.shape[:2]
            img_rgb = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
            img_tensor = torch.tensor(img_rgb / 255.0, dtype=torch.float32, device=self.device).permute(2, 0, 1).unsqueeze(0)
            
            # 自动判断 RealSense 深度尺度
            raw_depth_scale = 0.001 if (depth_img.dtype == np.uint16 or depth_img.max() > 1000) else 1.0
            depth_m = (depth_img.astype(np.float32) * raw_depth_scale)
            depth_m = np.nan_to_num(depth_m, nan=0.0, posinf=0.0, neginf=0.0)
            depth_tensor = torch.tensor(depth_m, dtype=torch.float32, device=self.device)
            
            # 归一化内参
            K_norm = self.intrinsics_np.copy()
            K_norm[0, 0] /= w; K_norm[0, 2] /= w
            K_norm[1, 1] /= h; K_norm[1, 2] /= h
            K_tensor = torch.tensor(K_norm, dtype=torch.float32, device=self.device).unsqueeze(0)

            with torch.no_grad():
                output = self.depth_model.infer(img_tensor, depth_in=depth_tensor, apply_mask=True, intrinsics=K_tensor)
                depth_refined_m = output['depth'].squeeze().cpu().numpy()

            # ---------------------------------------------------------
            # 5. 解析并遍历目标 (检测 + 测厚)
            # ---------------------------------------------------------
            detections: List[Dict[str, Any]] =[]
            result_img = color_img.copy()

            clss = boxes.cls.cpu().numpy().astype(int)
            confs = boxes.conf.cpu().numpy()
            polygons = masks.xy # 获取掩码多边形

            for i in range(len(clss)):
                class_id = clss[i]
                cf = float(confs[i])

                cls_entry = self.cls_map.get(class_id)
                if not cls_entry:
                    return {"analyse_result": "error", "message": f"推理返回未定义类别 {class_id}", "detections":[]}

                zh_name = cls_entry.get("name", f"类：{class_id}")
                en_code = cls_entry.get("code", f"class:{class_id}")
                name = zh_name if self.has_zh_font else en_code

                # 生成该单个目标的独占 Mask
                poly = polygons[i]
                if poly.shape[0] < 3:
                    continue
                obj_mask = np.zeros((h, w), dtype=np.uint8)
                pts = np.round(poly).astype(np.int32)
                cv2.fillPoly(obj_mask, [pts], 255)

                # ========= 核心：测厚并获取基于 Mask 最小外接矩形的 bbox =========
                thickness, rect_box_pts = self.measurer.calculate_thickness(result_img, depth_refined_m, obj_mask)
                
                # Bbox 处理：获取最小外接矩形对应的水平 [x, y, w, h]
                if rect_box_pts is not None:
                    rx, ry, rw, rh = cv2.boundingRect(rect_box_pts)
                else:
                    # 如果几何提取失败，回退使用 YOLO 原生的 bbox
                    x1f, y1f, x2f, y2f = boxes.xyxy[i].tolist()
                    rx, ry = int(x1f), int(y1f)
                    rw, rh = int(x2f - x1f), int(y2f - y1f)

                # 状态判定
                status = "测厚成功" if thickness else "测厚失败(几何异常)"
                final_thickness = float(thickness) if thickness else -1.0

                det = {
                    "abnormal_obj": False,
                    "type_code": en_code,
                    "type_name": zh_name,
                    "confidence": cf,
                    "bbox": [rx, ry, rw, rh],
                    "status": status,
                    "thickness": final_thickness  # 👈 新增厚度字段透传给 ROS
                }
                detections.append(det)

                # 绘制可视化信息
                color = self.palette[class_id % len(self.palette)]
                label_text = f"{name} {cf:.2f}"
                self._draw_label(result_img, (rx, ry, rx + rw, ry + rh), label_text, color)
                
                if thickness:
                    cv2.putText(result_img, f"Thick: {thickness:.1f}mm", (rx, ry - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # ---------------------------------------------------------
            # 6. 统计状态并生成 Message
            # ---------------------------------------------------------
            status_counts: Dict[str, int] = {}
            for d in detections:
                s = d.get("status") or "正常"
                status_counts[s] = status_counts.get(s, 0) + 1

            parts = [f"共{len(detections)}个目标"]
            for k, v in status_counts.items():
                parts.append(f"{v}个{k}")
            message = ','.join(parts)
            
            # 保存最终的可视化底图
            out_path = save_path_prefix + "_result.png"
            cv2.imwrite(out_path, result_img)
            
            # 打印日志 (ROS端可直接看到)
            point_id = os.path.basename(save_path_prefix)
            print(f"[ROS-Infer] 第{point_id}个点处理完成: {message}")

            return {"analyse_result": "ok", "message": message, "detections": detections}

        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"analyse_result": "error", "message": f"推理过程发生异常: {str(e)}", "detections": []}
        
if __name__ == "__main__":
    # 只需要传入配置文件路径
    config_path = resolve_path('$(find detect)/config/thickness_detector.yaml')
    node = ThicknessDetector(config_path)
    img_color = cv2.imread('/home/shansu/moma_ws/src/debug_tools/data/1/color_2.png')
    img_depth = cv2.imread('/home/shansu/moma_ws/src/debug_tools/data/1/depth_2.png', cv2.IMREAD_UNCHANGED)
    # 模拟图片输入
    # node.infer(...)
    node.infer('/home/shansu/moma_ws/records/1', img_color, img_depth)
