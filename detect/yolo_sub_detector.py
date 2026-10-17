#! /usr/bin/env python
# --- Standard library ---
import math
import os
from typing import Any, Dict, Mapping, Optional
from pathlib import Path

# --- Third-party ---
import cv2
import numpy as np
import yaml
import torch
from ultralytics import YOLO

try:
    from .path_utils import load_yaml_config, resolve_ros_substitutions
except ImportError:
    from path_utils import load_yaml_config, resolve_ros_substitutions

# import argparse
import json
import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any


# ===================== 全局模型加载 =====================

_sub_detector_model = None
_sub_detector_config = None
_status_classifier = None

# ===================== 状态规范化映射 =====================

STATUS_MAPPING = {
    "normal": "正常",
    "正常": "正常",
    "abnormal": "异常",
    "异常": "异常",
    "unpainted": "未画",
    "未画": "未画",
}


def _normalize_status(status: str) -> str:
    """将英文状态或中文状态统一转换为标准中文状态名称（正常/异常/未画）"""
    if not status:
        return "正常"
    s = str(status).strip().lower()
    return STATUS_MAPPING.get(s, str(status))


def _default_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "config" / "yolo_sub_detector.yaml"


def _resolve_config_path(value: str | Path) -> Path:
    path = Path(resolve_ros_substitutions(str(value))).expanduser()
    if path.is_absolute():
        return path
    return (_default_config_path().parent / path).resolve()

def _load_sub_detector_config():
    """加载子检测器配置文件"""
    global _sub_detector_config
    if _sub_detector_config is not None:
        return _sub_detector_config
    
    config_path = _default_config_path()
    try:
        _sub_detector_config = load_yaml_config(str(config_path))
        print(f"[SubDetector] 配置加载成功: {config_path}")
        return _sub_detector_config
    except Exception as e:
        print(f"[SubDetector] 配置加载失败: {e}")
        return None

def _load_sub_detector_model():
    """懒加载子检测器模型"""
    global _sub_detector_model, _sub_detector_config
    if _sub_detector_model is not None:
        return _sub_detector_model
    
    # 加载配置
    config = _load_sub_detector_config()
    if config is None:
        return None
    
    weights = config.get('weights')
    device = config.get('device', 'cuda')
    
    if not weights:
        print("[SubDetector] 配置中缺少 weights 字段")
        return None
    
    weights_path = _resolve_config_path(weights)
    if not weights_path.is_file():
        print(f"[SubDetector] 模型文件不存在: {weights_path}")
        return None
    
    try:
        _sub_detector_model = YOLO(str(weights_path))
        if device:
            _sub_detector_model.to(device)
        print(f"[SubDetector] 模型加载成功: {weights_path}")
        print(f"[SubDetector] 推理设备: {device}")
        return _sub_detector_model
    except Exception as e:
        print(f"[SubDetector] 模型加载失败: {e}")
        return None


# ===================== 数据结构 =====================

@dataclass
class MarkComponent:
    """表示一条防松漆连通区域（近似一条线段）"""
    comp_id: int
    contour: np.ndarray          # (N, 2) float32
    area: float
    center: Tuple[float, float]
    length: float                # minAreaRect 长边
    width: float                 # minAreaRect 短边（近似线宽）
    angle_deg: float             # 长边方向角（仅信息用）
    v: np.ndarray                # (2,) 长边方向单位向量
    n: np.ndarray                # (2,) 法向单位向量


@dataclass
class PairAnalysis:
    """一对线段之间的几何关系"""
    comp_i: int
    comp_j: int
    du: float                    # 沿线方向最近距离（gap）
    delta_n: float               # 法向偏移
    angle_diff_deg: float        # 方向差
    label: str                   # "aligned" / "misaligned" / "unknown"
    confidence: float            # 0~1
    extra: Dict[str, Any]        # 额外调试信息


@dataclass
class TorchMaskClassifier:
    model: Any
    class_names: List[str]
    device: Any

    def predict_scores(self, mask: np.ndarray, imgsz: int) -> np.ndarray:
        resized = cv2.resize(mask, (imgsz, imgsz), interpolation=cv2.INTER_NEAREST)
        inputs = torch.from_numpy((resized > 0).astype(np.float32))
        inputs = inputs.to(device=self.device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        with torch.inference_mode():
            probabilities = torch.softmax(self.model(inputs), dim=1)[0]
        return probabilities.detach().cpu().numpy()


def _build_tiny_mask_resnet(num_classes: int) -> Any:
    class ResidualBlock(torch.nn.Module):
        def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
            self.bn1 = torch.nn.BatchNorm2d(out_channels)
            self.conv2 = torch.nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
            self.bn2 = torch.nn.BatchNorm2d(out_channels)
            self.skip = (
                torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                    torch.nn.BatchNorm2d(out_channels),
                )
                if stride != 1 or in_channels != out_channels
                else torch.nn.Identity()
            )

        def forward(self, inputs: Any) -> Any:
            outputs = torch.relu(self.bn1(self.conv1(inputs)))
            outputs = self.bn2(self.conv2(outputs))
            return torch.relu(outputs + self.skip(inputs))

    class TinyMaskResNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = torch.nn.Sequential(
                torch.nn.Conv2d(1, 32, 3, 1, 1, bias=False),
                torch.nn.BatchNorm2d(32),
                torch.nn.ReLU(inplace=True),
            )
            self.features = torch.nn.Sequential(
                ResidualBlock(32, 32), ResidualBlock(32, 64, 2),
                ResidualBlock(64, 64), ResidualBlock(64, 128, 2),
                ResidualBlock(128, 128), ResidualBlock(128, 256, 2),
                ResidualBlock(256, 256),
            )
            self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)
            self.max_pool = torch.nn.AdaptiveMaxPool2d(1)
            self.classifier = torch.nn.Sequential(
                torch.nn.Flatten(), torch.nn.Dropout(0.25), torch.nn.Linear(512, num_classes)
            )

        def forward(self, inputs: Any) -> Any:
            features = self.features(self.stem(inputs))
            pooled = torch.cat((self.avg_pool(features), self.max_pool(features)), dim=1)
            return self.classifier(pooled)

    return TinyMaskResNet()


def _resolve_torch_device(device: Optional[str]) -> Any:
    requested = str(device).strip() if device is not None else ""
    value = requested or ("cuda:0" if torch.cuda.is_available() else "cpu")
    if value.isdigit():
        value = f"cuda:{value}"
    resolved = torch.device(value)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"请求了设备 {value}，但当前 PyTorch 无法使用 CUDA")
    return resolved


def _load_status_classifier() -> TorchMaskClassifier:
    global _status_classifier
    if _status_classifier is not None:
        return _status_classifier

    config = _load_sub_detector_config()
    section = config.get("status_classification", {}) if config else {}
    weights = section.get("weights")
    if not weights:
        raise ValueError("配置中缺少 status_classification.weights")
    weights_path = _resolve_config_path(weights)
    if not weights_path.is_file():
        raise FileNotFoundError(f"分类模型不存在: {weights_path}")

    checkpoint = torch.load(str(weights_path), map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, Mapping) or "model_state_dict" not in checkpoint:
        raise TypeError(f"{weights_path} 不是 TinyMaskResNet checkpoint")
    class_names = [str(name) for name in checkpoint.get("class_names", [])]
    if not class_names:
        raise ValueError(f"{weights_path} 缺少 class_names")

    model = _build_tiny_mask_resnet(len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    device = _resolve_torch_device(section.get("device"))
    model.to(device)
    model.eval()
    _status_classifier = TorchMaskClassifier(model, class_names, device)
    print(f"[SubDetector] 状态分类模型加载成功: {weights_path}")
    return _status_classifier


# ===================== 核心步骤 =====================

def extract_red_mask(
        img_bgr: np.ndarray,
        # lower_red1=(0, 150, 150),
        # upper_red1=(7, 240, 255),
        # lower_red2=(170, 80, 80),
        # upper_red2=(180, 255, 255),
        lower_red1=(0, 48, 63),
        upper_red1=(12, 255, 255),
        lower_red2=(170, 48, 63),
        upper_red2=(179, 255, 255),
        morph_kernel_size: int = 3,
        morph_iter: int = 1,
) -> np.ndarray:
    """
    提取红色防松漆的二值掩码。

    返回:
        mask (uint8): 0/255
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array(lower_red1, np.uint8)
    upper_red1 = np.array(upper_red1, np.uint8)
    lower_red2 = np.array(lower_red2, np.uint8)
    upper_red2 = np.array(upper_red2, np.uint8)

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)
    # mask = mask1

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size)
    )
    if morph_iter > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=morph_iter)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=morph_iter)

    return mask


def extract_purple_mask(
        img_bgr: np.ndarray,
        lower_purple=(130, 80, 80),
        upper_purple=(160, 255, 255),
        morph_kernel_size: int = 3,
        morph_iter: int = 1,
) -> np.ndarray:
    """
    提取紫色防松标识的二值掩码。

    返回:
        mask (uint8): 0/255
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    lower_purple = np.array(lower_purple, np.uint8)
    upper_purple = np.array(upper_purple, np.uint8)

    mask = cv2.inRange(hsv, lower_purple, upper_purple)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size)
    )
    if morph_iter > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=morph_iter)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=morph_iter)

    return mask


def find_components(
        mask: np.ndarray,
        min_area: float = 50.0,
        min_length: float = 10.0,
        min_slender_ratio: float = 2.0,
) -> List[MarkComponent]:
    """
    从二值 mask 中找出所有“像线”的连通域，并用 minAreaRect 拟合。

    过滤掉:
        - 面积太小的噪点
        - 长宽比不够细长的块（比如涂抹一团而不是线）
    """
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    components: List[MarkComponent] = []

    for cid, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        pts = cnt.reshape(-1, 2).astype(np.float32)
        rect = cv2.minAreaRect(pts)
        (cx, cy), (w, h), angle = rect

        if w <= 0 or h <= 0:
            continue

        # 取长边为 length，短边为 width
        if w >= h:
            length, width = float(w), float(h)
            angle_dir = angle
        else:
            length, width = float(h), float(w)
            angle_dir = angle + 90.0  # OpenCV 角度定义: 让 angle_dir 对应长边方向

        if length < min_length:
            continue

        slender_ratio = length / max(width, 1e-6)
        if slender_ratio < min_slender_ratio:
            # 太“圆/块”了，不像一条线
            continue

        # 用 boxPoints 取长边方向向量
        box = cv2.boxPoints(((cx, cy), (length, width), angle_dir))
        box = box.astype(np.float32)
        # 选两条相邻边中较长的一条作为长边
        e01 = box[1] - box[0]
        e12 = box[2] - box[1]
        if np.linalg.norm(e01) >= np.linalg.norm(e12):
            edge = e01
        else:
            edge = e12

        v = edge / (np.linalg.norm(edge) + 1e-8)
        n_vec = np.array([-v[1], v[0]], dtype=np.float32)

        comp = MarkComponent(
            comp_id=cid,
            contour=pts,
            area=float(area),
            center=(float(cx), float(cy)),
            length=float(length),
            width=float(width),
            angle_deg=float(angle_dir),
            v=v,
            n=n_vec,
        )
        components.append(comp)

    return components


def _pair_u_n_stats(
        comp: MarkComponent,
        v_pair: np.ndarray,
        n_pair: np.ndarray,
) -> Dict[str, float]:
    """
    将某个 component 的 contour 点投影到给定 (v_pair, n_pair) 坐标系，返回 u/n 范围和均值。
    """
    pts = comp.contour  # (N, 2)
    u = pts @ v_pair      # (N,)
    n = pts @ n_pair      # (N,)

    return {
        "u_min": float(u.min()),
        "u_max": float(u.max()),
        "u_mean": float(u.mean()),
        "n_min": float(n.min()),
        "n_max": float(n.max()),
        "n_mean": float(n.mean()),
        "n_std": float(n.std() + 1e-6),
    }


def fuzzy_label_from_delta_n(delta_n: float,
                             align_thr: float,
                             misalign_thr: float) -> Tuple[str, float, Dict[str, float]]:
    """
    根据 delta_n, align_thr, misalign_thr 计算
    aligned / unknown / misaligned 的 fuzzy 概率，并返回 label 和置信度。
    目前退化成阈值法。
    返回:
        label: str
        confidence: float
        probs: dict {aligned, unknown, misaligned}
    """
    x = float(delta_n)
    a = float(max(align_thr, 1e-6))
    m = float(max(misalign_thr, a + 1e-6))

    # μ_align
    if x <= 0:
        mu_align = 1.0
    elif x < a:
        mu_align = 1.0 - x / a
    else:
        mu_align = 0.0

    # μ_mis: 梯形，顶点在m
    mid = (a + m) / 2.0
    if x <= mid:
        mu_mis = 0.0
    elif x < m:
        mu_mis = max(0.0, (x - mid) / (m - mid))
    else:
        mu_mis = 1.0

    # μ_unknown: 大三角形，峰值在 mid
    if x <= 0:
        mu_unk = 0.0
    elif x < mid:
        mu_unk = x / mid
    elif x < m:
        mu_unk = max(0.0, 1.0 - (x - mid) / (m - mid))
    else:
        mu_unk = 0.0

    sum_mu = mu_align + mu_unk + mu_mis
    if sum_mu <= 1e-8:
        # 极端异常情况下，全部当 unknown 处理
        probs = dict(aligned=0.0, unknown=1.0, misaligned=0.0)
        return "unknown", 1.0, probs

    p_align = mu_align / sum_mu
    p_unk   = mu_unk   / sum_mu
    p_mis   = mu_mis   / sum_mu

    probs = dict(aligned=p_align, unknown=p_unk, misaligned=p_mis)
    label = max(probs, key=probs.get)
    confidence = float(probs[label])
    return label, confidence, probs


def analyze_pairs(
        components: List[MarkComponent],
        max_center_dist_factor: float = 3.0,
        max_angle_diff_deg: float = 25.0,
        du_factor: float = 3.0,
        align_tol_factor: float = 0.8 ,
        misalign_tol_factor: float = 1.4,
) -> List[PairAnalysis]:
    """
    对所有线段两两配对，找出可能的“关节”，并判断是否对齐。

    策略:
        1) 先按中心距离和方向差做粗过滤;
        2) 再用投影到 pair 平均方向的 u/n 计算 du 和 Δn;
        3) du 小 → 有可能是一对关节线，Δn 决定 aligned / misaligned。

    返回:
        PairAnalysis 列表，每个代表一个“关节 pair”。
    """
    pairs: List[PairAnalysis] = []
    n_comp = len(components)
    if n_comp < 2:
        return pairs

    # 所有线段长度的中位数，用于尺度估计
    lengths = np.array([c.length for c in components], dtype=np.float32)
    median_len = float(np.median(lengths))

    for i in range(n_comp):
        ci = components[i]
        for j in range(i + 1, n_comp):
            cj = components[j]

            # 1) 粗过滤: 中心距离
            dx = ci.center[0] - cj.center[0]
            dy = ci.center[1] - cj.center[1]
            center_dist = math.hypot(dx, dy)

            # 同一 group 的线段距离大致不会超过 若干个 length
            max_center_dist = max_center_dist_factor * max(ci.length, cj.length, median_len)
            if center_dist > max_center_dist:
                continue

            # 2) 方向差: 使用两条方向向量的夹角
            dot = float(np.clip(np.dot(ci.v, cj.v), -1.0, 1.0))
            angle_diff_deg = math.degrees(math.acos(abs(dot)))  # 不关心正反方向
            if angle_diff_deg > max_angle_diff_deg:
                continue

            # 3) 以平均方向作为 pair 的“主方向”
            v_pair = ci.v + cj.v
            if np.linalg.norm(v_pair) < 1e-3:
                v_pair = ci.v
            v_pair = v_pair / (np.linalg.norm(v_pair) + 1e-8)
            n_pair = np.array([-v_pair[1], v_pair[0]], dtype=np.float32)

            stats_i = _pair_u_n_stats(ci, v_pair, n_pair)
            stats_j = _pair_u_n_stats(cj, v_pair, n_pair)

            umin_i, umax_i = stats_i["u_min"], stats_i["u_max"]
            umin_j, umax_j = stats_j["u_min"], stats_j["u_max"]

            # du: 两个 u 区间之间的“gap”（如果重叠则为 0）
            du = max(0.0, max(umin_i, umin_j) - min(umax_i, umax_j))

            # 平均线宽
            thickness = (ci.width + cj.width) / 2.0
            du_thr = du_factor * thickness + 1.0

            if du > du_thr:
                # 沿线方向离得太远，认为不是同一个关节
                continue

            # Δn: 法向偏移
            n_mean_i = stats_i["n_mean"]
            n_mean_j = stats_j["n_mean"]
            delta_n = abs(n_mean_i - n_mean_j)

            align_thr = align_tol_factor * thickness  # 认为“几乎在一条线”的容差
            misalign_thr = misalign_tol_factor * thickness  # 认为“明显错开”的阈值

            # # 判定
            # label = "unknown"
            # confidence = 0.0
            #
            # if delta_n <= align_thr:
            #     label = "aligned"
            #     # Δn 越小置信度越高
            #     if align_thr > 1e-6:
            #         confidence = 0.3 + 0.7 * max(0.0, 1.0 - delta_n / align_thr)
            #     else:
            #         confidence = 0.5
            # elif delta_n >= misalign_thr:
            #     label = "misaligned"
            #     # Δn 越大(相对线宽)置信度越高，但截断到 0.99
            #     ratio = delta_n / misalign_thr
            #     confidence = float(max(0.3, min(0.99, ratio)))
            # else:
            #     label = "unknown"
            #     confidence = 0.4

            label, confidence, probs = fuzzy_label_from_delta_n(
                delta_n=delta_n,
                align_thr=align_thr,
                misalign_thr=misalign_thr,
            )

            pair = PairAnalysis(
                comp_i=ci.comp_id,
                comp_j=cj.comp_id,
                du=float(du),
                delta_n=float(delta_n),
                angle_diff_deg=float(angle_diff_deg),
                label=label,
                confidence=float(confidence),
                extra=dict(
                    thickness=float(thickness),
                    center_dist=float(center_dist),
                    u_stats_i=stats_i,
                    u_stats_j=stats_j,
                    probs=probs,
                ),
            )
            pairs.append(pair)

    return pairs


# ===================== 可视化 =====================

def visualize_components_mask(
        img_shape: Tuple[int, int, int],
        components: List[MarkComponent],
) -> np.ndarray:
    """
    绘制“彩色掩码图”：每个 component 用不同颜色填充。

    返回:
        彩色 BGR 图像，背景为黑色。
    """
    h, w, _ = img_shape
    canvas = np.zeros((h, w, 3), dtype=np.uint8)

    rng = random.Random(2025)

    for comp in components:
        color = (
            rng.randint(64, 255),
            rng.randint(64, 255),
            rng.randint(64, 255),
        )
        cnt = comp.contour.astype(np.int32).reshape(-1, 1, 2)
        cv2.drawContours(canvas, [cnt], contourIdx=-1, color=color, thickness=-1)

    return canvas


def visualize_on_original(
        img_bgr: np.ndarray,
        components: List[MarkComponent],
        pairs: List[PairAnalysis],
        draw_pairs: bool = True,
) -> np.ndarray:
    """
    在原图上叠加:
        - 每个 component 的 minAreaRect 矩形 + 方向箭头
        - 每个 pair 的连线 & 文本标签(du, Δn, label)
    """
    vis = img_bgr.copy()
    h, w = vis.shape[:2]

    if len(components) == 0:
        cv2.putText(vis,
                    "No components found",
                    (10, min(30, h - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                    lineType=cv2.LINE_AA)
        return vis

    # 先画 component 的外框和编号
    for comp in components:
        text_bias = int(comp.width)
        rect = ((comp.center[0], comp.center[1]),
                (comp.length, comp.width),
                comp.angle_deg)
        box = cv2.boxPoints(rect).astype(np.int32)
        cv2.polylines(vis, [box], isClosed=True, color=(0, 255, 255), thickness=2)

        cx, cy = comp.center
        cv2.putText(vis,
                    f"id={comp.comp_id}",
                    (int(cx) - text_bias, int(cy) - text_bias),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    1,
                    lineType=cv2.LINE_AA)

    # 再画 pair 信息
    if draw_pairs:
        for p in pairs:
            text_bias = int(p.extra["thickness"]) * 2
            ci = next(c for c in components if c.comp_id == p.comp_i)
            cj = next(c for c in components if c.comp_id == p.comp_j)

            cxi, cyi = map(int, ci.center)
            cxj, cyj = map(int, cj.center)

            if p.label == "aligned":
                color = (0, 255, 0)
            elif p.label == "misaligned":
                color = (0, 0, 255)
            else:
                color = (255, 255, 0)

            cv2.line(vis, (cxi, cyi), (cxj, cyj), color=color, thickness=2)

            mx, my = (cxi + cxj) // 2, (cyi + cyj) // 2
            txt = f"{p.label}, du={p.du:.1f}, dn={p.delta_n:.1f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.45
            thickness = 1
            (txt_w, txt_h), baseline = cv2.getTextSize(txt, font, font_scale, thickness)

            tx = max(4, min(w - txt_w - 4, mx - text_bias))
            ty = max(txt_h + 6, min(h - baseline - 4, my + text_bias))

            cv2.rectangle(
                vis,
                (tx - 3, ty - txt_h - 3),
                (tx + txt_w + 3, ty + baseline + 3),
                (0, 0, 0),
                thickness=-1,
            )
            cv2.putText(vis,
                        txt,
                        (tx, ty),
                        font,
                        font_scale,
                        color,
                        thickness,
                        lineType=cv2.LINE_AA)

    if len(pairs) == 0:
        cv2.putText(vis,
                    "No valid pairs",
                    (10, min(60, h - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                    lineType=cv2.LINE_AA)

    # 最后画方向箭头，避免被 pair 连线覆盖
    for comp in components:
        cx, cy = comp.center
        arrow_len = max(comp.length * 0.6, 18.0)
        start_pt = (
            max(0, min(w - 1, int(round(cx)))),
            max(0, min(h - 1, int(round(cy)))),
        )
        end_pt = (
            max(0, min(w - 1, int(round(cx + comp.v[0] * arrow_len)))),
            max(0, min(h - 1, int(round(cy + comp.v[1] * arrow_len)))),
        )

        cv2.circle(vis, start_pt, 3, (255, 255, 255), thickness=-1)
        cv2.arrowedLine(vis,
                        start_pt,
                        end_pt,
                        color=(0, 0, 0),
                        thickness=5,
                        tipLength=0.32)
        cv2.arrowedLine(vis,
                        start_pt,
                        end_pt,
                        color=(255, 0, 0),
                        thickness=3,
                        tipLength=0.32)

    return vis


# ===================== 总入口函数 =====================

def check_alignment(
        img_bgr: np.ndarray,
        red_mask_params: Dict[str, Any] = None,
        comp_params: Dict[str, Any] = None,
        pair_params: Dict[str, Any] = None,
) -> Tuple[np.ndarray, List[MarkComponent], List[PairAnalysis], np.ndarray]:
    """
    对整张 BGR 图像进行防松漆对齐分析。

    返回:
        mask_red       : 二值红漆 mask
        components     : 连通线段列表
        pairs          : 关节 pair 列表
        comp_mask_vis  : 彩色组件掩码图
    """
    if red_mask_params is None:
        red_mask_params = {}
    if comp_params is None:
        comp_params = {}
    if pair_params is None:
        pair_params = {}

    mask_red = extract_red_mask(img_bgr, **red_mask_params)
    components = find_components(mask_red, **comp_params)
    pairs = analyze_pairs(components, **pair_params)
    comp_mask_vis = visualize_components_mask(img_bgr.shape, components)

    return mask_red, components, pairs, comp_mask_vis


# ===================== CLI 示例 =====================

def _check_anti_loosen_mark(img: np.ndarray, save_path_prefix: str):
    """使用 YOLO 分割和掩码分类模型，保持旧的二元返回协议。"""
    try:
        if img is None or not isinstance(img, np.ndarray) or img.size == 0:
            raise ValueError("子检测器输入图像异常")
        config = _load_sub_detector_config()
        segmentation = config.get("segmentation", {}) if config else {}
        model = _load_sub_detector_model()
        classifier = _load_status_classifier()
        if model is None:
            raise RuntimeError("分割模型加载失败")

        results = model.predict(
            source=img,
            imgsz=int(segmentation.get("imgsz", config.get("imgsz", 640))),
            conf=float(segmentation.get("confidence", config.get("confidence", 0.4))),
            iou=float(segmentation.get("iou", config.get("iou", 0.3))),
            verbose=False,
            device=segmentation.get("device") or config.get("device") or None,
        )
    except Exception as e:
        print(f"[SubDetector] 数据驱动防松标识检测异常: {e}")
        return "异常", "error"

    result = results[0] if results else None
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    target_classes = {
        str(value).strip().casefold()
        for value in segmentation.get("mask_target_classes", ["anti_loosen_mark"])
    }
    masks = getattr(result, "masks", None) if result is not None else None
    mask_data = getattr(masks, "data", None) if masks is not None else None
    boxes = getattr(result, "boxes", None) if result is not None else None
    class_ids = boxes.cls.detach().cpu().numpy().astype(int) if boxes is not None else []
    names = getattr(model, "names", {})
    if mask_data is not None:
        instances = mask_data.detach().cpu().numpy()
        if instances.ndim == 2:
            instances = instances[None, ...]
        for index, instance in enumerate(instances):
            class_id = int(class_ids[index]) if index < len(class_ids) else -1
            class_name = str(names.get(class_id, class_id)).casefold()
            if str(class_id) not in target_classes and class_name not in target_classes:
                continue
            instance_mask = np.where(instance > 0.5, 255, 0).astype(np.uint8)
            if instance_mask.shape != mask.shape:
                instance_mask = cv2.resize(
                    instance_mask,
                    (mask.shape[1], mask.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
            mask = cv2.bitwise_or(mask, instance_mask)

    scores = classifier.predict_scores(mask, int(config["status_classification"].get("imgsz", 96)))
    raw_status = classifier.class_names[int(np.argmax(scores))]
    model_status = _normalize_status(raw_status)
    confidence = float(np.max(scores))
    cv2.imwrite(f"{save_path_prefix}_mask_from_yolo.png", mask)
    with open(f"{save_path_prefix}_status.json", "w", encoding="utf-8") as file:
        json.dump({"status": model_status, "raw_status": raw_status, "confidence": confidence}, file, ensure_ascii=False, indent=2)

    return model_status, "ok"


def _check_anti_loosen_mark_geometry(img: np.ndarray, save_path_prefix: str):
    """
    检查防松标识是否对齐。
    流程：YOLO 分割模型 → 提取 mask → find_components → analyze_pairs → 几何判断
    
    Args:
        img: np.ndarray (BGR, uint8, crop 的检测框区域)
        save_path_prefix: 保存输出文件的前缀
    
    Returns:
        (status: str, result: "ok" or "error")
        status 可能值: "正常", "错位", "未画", "异常"
    """
    try:
        # ==================== Step 1: 用 YOLO 分割模型进行推理 ====================
        model = _load_sub_detector_model()
        if model is None:
            print("[SubDetector] 模型加载失败，降级到传统方法")
            return _check_anti_loosen_mark_legacy(img, save_path_prefix)
        
        config = _load_sub_detector_config()
        if config is None:
            print("[SubDetector] 配置加载失败，降级到传统方法")
            return _check_anti_loosen_mark_legacy(img, save_path_prefix)
        
        imgsz = config.get('imgsz', 640)
        conf = config.get('confidence', 0.4)
        iou = config.get('iou', 0.3)
        device = config.get('device', 'cuda')
        
        # 执行分割推理
        results = model.predict(
            source=img,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            verbose=False,
            device=device,
        )
        
        if not results:
            print("[SubDetector] 推理未返回结果")
            return "未画", "error"
        
        res = results[0]
        
        # ==================== Step 2: 从分割结果提取 mask ====================
        masks = getattr(res, "masks", None)
        if masks is None or len(masks) == 0:
            print("[SubDetector] 分割模型未检测到防松标识")
            return "未画", "error"
        
        # 如果有多个 mask，合并为一个二值图
        # masks.data 形状: (N, H, W)，N 为检测个数
        mask_data = masks.data.cpu().numpy()  # (N, H, W)
        if len(mask_data.shape) == 3:
            # 合并所有 masks：取并集（或运算）
            combined_mask = np.any(mask_data, axis=0).astype(np.uint8) * 255
        else:
            combined_mask = (mask_data > 0.5).astype(np.uint8) * 255

        img_h, img_w = img.shape[:2]
        mask_h, mask_w = combined_mask.shape[:2]
        print(f"[SubDetector] crop shape={img.shape}, mask shape={combined_mask.shape}")

        if (mask_h, mask_w) != (img_h, img_w):
            print(f"[SubDetector] mask 尺寸与 crop 不一致，执行对齐: mask=({mask_h}, {mask_w}) -> img=({img_h}, {img_w})")
            combined_mask = cv2.resize(combined_mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)

        nonzero_pixels = int(np.count_nonzero(combined_mask))
        print(f"[SubDetector] mask 非零像素数: {nonzero_pixels}")

        # 保存 YOLO 推理结果图
        res_plotted = res.plot()
        cv2.imwrite(f"{save_path_prefix}_yolo_det.png", res_plotted)
        cv2.imwrite(f"{save_path_prefix}_mask_from_yolo.png", combined_mask)

        mask_overlay = img.copy()
        if nonzero_pixels > 0:
            overlay_color = np.array([0, 255, 255], dtype=np.uint8)
            mask_idx = combined_mask > 0
            mask_overlay[mask_idx] = (
                0.4 * mask_overlay[mask_idx] + 0.6 * overlay_color
            ).astype(np.uint8)
        cv2.imwrite(f"{save_path_prefix}_mask_overlay.png", mask_overlay)
        
        # ==================== Step 3: 用 mask 进行后续的几何分析 ====================
        # 直接使用默认参数，避免在调用处重复维护同一组阈值
        components = find_components(combined_mask)
        pairs = analyze_pairs(components)
        comp_mask_vis = visualize_components_mask(img.shape, components)

        print(f"[SubDetector] 提取到 {len(components)} 个 components, {len(pairs)} 个 pairs")
        for comp in components:
            print(
                f"[SubDetector] component id={comp.comp_id}, center=({comp.center[0]:.1f}, {comp.center[1]:.1f}), "
                f"length={comp.length:.1f}, width={comp.width:.1f}, angle={comp.angle_deg:.1f}"
            )
        for idx, p in enumerate(pairs):
            print(
                f"[SubDetector] pair {idx}: label={p.label}, du={p.du:.2f}, dn={p.delta_n:.2f}, "
                f"angle_diff={p.angle_diff_deg:.2f}, probs={p.extra.get('probs', {})}"
            )
        
        # ==================== Step 4: 几何判断和可视化 ====================
        vis_orig = visualize_on_original(img, components, pairs)
        
        # 保存调试图
        cv2.imwrite(f"{save_path_prefix}_mask_components.png", comp_mask_vis)
        cv2.imwrite(f"{save_path_prefix}_orig_debug.png", vis_orig)
        
        # 判断对齐状态
        pair_dicts = []
        status = "正常"
        for idx, p in enumerate(pairs):
            pair_dicts.append(
                dict(
                    pair_id=idx,
                    comp_i=p.comp_i,
                    comp_j=p.comp_j,
                    du=p.du,
                    delta_n=p.delta_n,
                    angle_diff_deg=p.angle_diff_deg,
                    label=p.label,
                    confidence=p.confidence,
                    extra=p.extra,
                )
            )
            # 如果有显著的错位 pair，标记为错位
            if p.extra['probs']["misaligned"] > 0.9:
                status = "错位"
        
        print(f"[SubDetector] 检测到 {len(components)} 条防松标识线，{len(pairs)} 个 pair，状态: {status}")
        
        if len(components) == 0:
            return "未画", "error"
        
        return status, "ok"
    
    except Exception as e:
        print(f'[SubDetector] 异常: {str(e)}')
        return "异常", "error"

def _check_anti_loosen_mark_legacy(img: np.ndarray, save_path_prefix: str):
    """
    降级方案：使用红色色彩分割方法（当 YOLO 模型不可用时使用）
    """
    try:
        # 使用各函数定义中的默认参数，避免在这里重复配置
        mask_red = extract_red_mask(img)
        components = find_components(mask_red)
        pairs = analyze_pairs(components)
        comp_mask_vis = visualize_components_mask(img.shape, components)

        vis_orig = visualize_on_original(img, components, pairs)
        
        cv2.imwrite(f"{save_path_prefix}_mask_red.png", mask_red)
        cv2.imwrite(f"{save_path_prefix}_mask_components.png", comp_mask_vis)
        cv2.imwrite(f"{save_path_prefix}_orig_debug.png", vis_orig)

        print(f"[SubDetector-Legacy] 提取到 {len(components)} 个 components, {len(pairs)} 个 pairs")
        for comp in components:
            print(
                f"[SubDetector-Legacy] component id={comp.comp_id}, center=({comp.center[0]:.1f}, {comp.center[1]:.1f}), "
                f"length={comp.length:.1f}, width={comp.width:.1f}, angle={comp.angle_deg:.1f}"
            )
        for idx, p in enumerate(pairs):
            print(
                f"[SubDetector-Legacy] pair {idx}: label={p.label}, du={p.du:.2f}, dn={p.delta_n:.2f}, "
                f"angle_diff={p.angle_diff_deg:.2f}, probs={p.extra.get('probs', {})}"
            )

        pair_dicts = []
        status = "正常"
        for idx, p in enumerate(pairs):
            pair_dicts.append(
                dict(
                    pair_id=idx,
                    comp_i=p.comp_i,
                    comp_j=p.comp_j,
                    du=p.du,
                    delta_n=p.delta_n,
                    angle_diff_deg=p.angle_diff_deg,
                    label=p.label,
                    confidence=p.confidence,
                    extra=p.extra,
                )
            )
            if p.extra['probs']["misaligned"] > 0.9:
                status = "错位"
        
        if len(components) == 0:
            return "未画", "error"
        return status, "ok"
    except Exception as e:
        print(f'[SubDetector] 降级方案异常: {str(e)}')
        return "异常", "error"
