import os
import re
import json
import cv2
import torch
import yaml
import numpy as np
from dataclasses import dataclass
from collections.abc import Mapping
from typing import Any, Dict, List, Optional, Tuple
from paddleocr import PaddleOCR
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
from PIL import Image, ImageDraw, ImageFont

try:
    from .path_utils import load_yaml_config, resolve_path
except ImportError:
    from path_utils import load_yaml_config, resolve_path


@dataclass
class _OCRItem:
    index: int
    text: str
    score: float
    box: np.ndarray
    center: Tuple[float, float]
    height: float


@dataclass
class _KeyLineCandidate:
    kind: str
    detected_text: str
    detected_score: float
    preprocess_variant: str
    box: np.ndarray
    crop: np.ndarray
    detected_content_end_x: int


def _to_gray(image: np.ndarray) -> np.ndarray:
    if image is None or image.size == 0:
        raise ValueError("待预处理图像为空")
    if image.ndim == 2:
        return image.astype(np.uint8, copy=True)
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    raise ValueError(f"不支持的图像形状：{image.shape}")


def _to_bgr(gray: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _fit_odd_kernel(requested: int, image_shape: Tuple[int, ...]) -> int:
    maximum = max(3, min(image_shape[:2]) - 1)
    if maximum % 2 == 0:
        maximum -= 1
    kernel = min(int(requested), maximum)
    if kernel % 2 == 0:
        kernel -= 1
    return max(kernel, 3)


def _percentile_stretch(gray: np.ndarray, low_percentile: float, high_percentile: float) -> np.ndarray:
    low, high = np.percentile(gray, [low_percentile, high_percentile])
    if high - low < 1.0:
        return gray.copy()
    stretched = (gray.astype(np.float32) - float(low)) * (255.0 / float(high - low))
    return np.clip(stretched, 0, 255).astype(np.uint8)


def _enhance_metal(gray: np.ndarray, config: Dict[str, Any]) -> np.ndarray:
    stretched = _percentile_stretch(gray, float(config["clip_low_percentile"]), float(config["clip_high_percentile"]))
    denoised = cv2.bilateralFilter(
        stretched,
        d=int(config["bilateral_diameter"]),
        sigmaColor=float(config["bilateral_sigma_color"]),
        sigmaSpace=float(config["bilateral_sigma_space"]),
    )
    kernel = _fit_odd_kernel(int(config["background_kernel"]), denoised.shape)
    background = np.maximum(cv2.GaussianBlur(denoised, (kernel, kernel), 0), 1)
    normalized = _percentile_stretch(cv2.divide(denoised, background, scale=192), 0.5, 99.5)
    clahe = cv2.createCLAHE(
        clipLimit=float(config["clahe_clip_limit"]),
        tileGridSize=(int(config["clahe_grid_size"]), int(config["clahe_grid_size"])),
    )
    contrasted = clahe.apply(normalized)
    amount = float(config["unsharp_amount"])
    if amount > 0:
        soft = cv2.GaussianBlur(contrasted, (0, 0), 1.0)
        contrasted = cv2.addWeighted(contrasted, 1.0 + amount, soft, -amount, 0)
    return contrasted


def _build_preprocess_variants(image: np.ndarray, config: Dict[str, Any]) -> Dict[str, np.ndarray]:
    mode = str(config["mode"]).lower()
    if mode not in {"none", "gray", "metal", "auto"}:
        raise ValueError(f"不支持的预处理模式：{mode}")
    if mode == "none":
        return {"none": image.copy()}
    gray = _to_gray(image)
    if mode == "gray":
        return {"gray": _to_bgr(gray)}
    metal = _to_bgr(_enhance_metal(gray, config))
    if mode == "metal":
        return {"metal": metal}
    return {"none": image.copy(), "gray": _to_bgr(gray), "metal": metal}


def _unwrap_result_mapping(result: Any) -> Mapping[str, Any]:
    if isinstance(result, Mapping):
        data = result
    else:
        json_value = getattr(result, "json", None)
        json_value = json_value() if callable(json_value) else json_value
        data = json.loads(json_value) if isinstance(json_value, str) else json_value
        if not isinstance(data, Mapping):
            to_dict = getattr(result, "to_dict", None)
            data = to_dict() if callable(to_dict) else None
    if not isinstance(data, Mapping):
        raise TypeError(f"不支持的 OCR 结果类型：{type(result).__name__}")
    for wrapper_key in ("res", "result"):
        wrapped = data.get(wrapper_key)
        if isinstance(wrapped, Mapping):
            return wrapped
    return data


def _materialize_prediction(raw_results: Any) -> List[Any]:
    return raw_results if isinstance(raw_results, (list, tuple)) else list(raw_results)


def _parse_ocr_items(raw_results: Any, min_score: float) -> List[_OCRItem]:
    results = _materialize_prediction(raw_results)
    if not results:
        return []
    data = _unwrap_result_mapping(results[0])
    required = ("dt_polys", "rec_texts", "rec_scores")
    missing = [key for key in required if key not in data]
    if missing:
        raise KeyError(f"OCR结果缺少关键键: {', '.join(missing)}")
    polygons, texts, scores = data["dt_polys"], data["rec_texts"], data["rec_scores"]
    if not (len(polygons) == len(texts) == len(scores)):
        raise ValueError("OCR结果数据长度不匹配")
    items = []
    for index, (polygon, text, score) in enumerate(zip(polygons, texts, scores), start=1):
        numeric_score = float(score)
        box = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
        if numeric_score < min_score or box.shape != (4, 2):
            continue
        items.append(_OCRItem(
            index=index, text=str(text), score=numeric_score, box=box,
            center=(float(box[:, 0].mean()), float(box[:, 1].mean())),
            height=max(float(np.ptp(box[:, 1])), 1.0),
        ))
    return items


def _suppress_large_relief_blobs(detail: np.ndarray) -> np.ndarray:
    positive = detail[detail > 0]
    if positive.size == 0:
        return detail.copy()
    threshold = max(float(np.percentile(positive, 85.0)), 1.0)
    labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (detail >= threshold).astype(np.uint8), connectivity=8
    )
    cleaned = detail.copy()
    height, width = detail.shape[:2]
    maximum_area = max(int(height * width * 0.025), 64)
    for label in range(1, labels_count):
        _, _, component_width, component_height, area = stats[label]
        if area > maximum_area or (component_width > height * 0.8 and component_height > height * 0.45):
            cleaned[labels == label] = 0
    return cleaned


def _build_local_relief_variants(image: np.ndarray) -> Dict[str, np.ndarray]:
    gray = _to_gray(image)
    if min(gray.shape[:2]) < 3:
        return {"local_gray": _to_bgr(gray)}
    base = cv2.bilateralFilter(gray, d=3, sigmaColor=25, sigmaSpace=25)
    kernel_size = max(5, min(31, int(round(base.shape[0] * 0.18))))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    variants = {"local_gray": _to_bgr(base)}
    for name, operation in (("local_dark_relief", cv2.MORPH_BLACKHAT), ("local_bright_relief", cv2.MORPH_TOPHAT)):
        response = _suppress_large_relief_blobs(cv2.morphologyEx(base, operation, kernel))
        positive = response[response > 0]
        if positive.size == 0:
            enhanced = base.copy()
        else:
            cap = max(float(np.percentile(positive, 99.0)), 1.0)
            accent = np.clip(response.astype(np.float32) * (72.0 / cap), 0, 72)
            enhanced = np.clip(base.astype(np.float32) - accent, 0, 255).astype(np.uint8)
        variants[name] = _to_bgr(enhanced)
    return variants


def _classify_key_line(text: str) -> Optional[str]:
    compact = re.sub(r"[\s:：]", "", str(text))
    if re.search(r"(?<![a-z])no\s*\.?", str(text), flags=re.IGNORECASE):
        return "No."
    if "型号" in compact or compact == "型" or (compact.startswith("号") and "编号" not in compact):
        return "型号"
    return None


def _parse_key_field(kind: str, text: str) -> Dict[str, Any]:
    compact = re.sub(r"\s+", "", str(text)).upper().replace("：", ":").replace("–", "-").replace("—", "-").replace("－", "-")
    if kind == "No.":
        match = re.fullmatch(r"([0-9])([A-Z])", compact) or re.search(r"NO\.?[:]?(?:\s*)([0-9])([A-Z])(?:$|[^A-Z0-9])", compact)
        value = f"{match.group(1)}{match.group(2)}" if match else None
        return {"value": value, "format_valid": match is not None, "complete": match is not None}
    base = "YZ-JXH-J" if "YZ-JXH-J" in compact else None
    match = re.search(r"YZ-JXH-J[:]?(?:\s*)([0-9](?:-HW)?)(?:$|[^A-Z0-9-])", compact) if base else None
    suffix = match.group(1) if match else None
    return {"value": f"{base} {suffix}" if suffix else base, "base_model": base, "suffix": suffix, "format_valid": match is not None, "complete": match is not None}


def _extract_yunda_suffix(text: str) -> Optional[str]:
    compact = re.sub(r"\s+", "", str(text)).upper().replace("–", "-").replace("—", "-").replace("－", "-").strip(".,:：;；|[]()")
    match = re.fullmatch(r"([0-9](?:-?HW)?)", compact)
    if match is None:
        return None
    value = match.group(1)
    return f"{value[0]}-{value[1:]}" if len(value) > 1 and "-" not in value else value


class OCRDetector:
    def __init__(self, config_path):
        """初始化OCR检测器，加载配置和模型"""
        self.config = self._load_config(config_path)
        self.preprocess_config = self._get_preprocess_config()
        self.key_line_config = self.config.get('key_line_recovery', {})
        self._line_recognizer = None
        self._last_no_code = None
        self._last_key_fields = {}
        self.ocr_model = self._init_ocr_model()
        self.yolo_model = self._init_yolo_model()
        # 加载用于中文绘制的字体
        self.font, self.has_zh_font = self._load_font()
        self.device = self.config['device'] or "cuda" if torch.cuda.is_available() else "cpu"

    def _load_config(self, config_path):
        """加载配置文件"""
        return load_yaml_config(config_path)

    def _get_preprocess_config(self) -> Dict[str, Any]:
        configured = self.config.get('preprocessing', {})
        defaults = {
            'mode': 'auto', 'min_score': 0.0,
            'clip_low_percentile': 1.0, 'clip_high_percentile': 99.0,
            'bilateral_diameter': 5, 'bilateral_sigma_color': 35.0,
            'bilateral_sigma_space': 35.0, 'clahe_clip_limit': 2.5,
            'clahe_grid_size': 8, 'background_kernel': 31,
            'unsharp_amount': 0.65, 'auto_quality_tie_ratio': 0.0,
        }
        defaults.update(configured)
        if not 0.0 <= float(defaults['min_score']) <= 1.0:
            raise ValueError('preprocessing.min_score 必须在 0 到 1 之间')
        return defaults
    
    def _init_ocr_model(self):
        """初始化OCR模型"""
        ocr_config = self.config['model']['ocr']
        return PaddleOCR(
            use_textline_orientation=ocr_config['use_angle_cls'],
            # lang=ocr_config['lang'],
            doc_unwarping_model_dir=ocr_config['UVDoc'],
            text_detection_model_dir=ocr_config['text_detection_model_dir'],
            text_recognition_model_dir=ocr_config['text_recognition_model_dir'],
            doc_orientation_classify_model_dir=ocr_config['doc_orientation_classify_model_dir'],
            textline_orientation_model_dir=ocr_config['textline_orientation_model_dir']
        )

    def _init_yolo_model(self):
        """初始化YOLO模型"""
        yolo_config = self.config['model']['yolo']
        if not os.path.exists(yolo_config['model_path']):
            raise FileNotFoundError(f"YOLO模型文件不存在: {yolo_config['model_path']}")
        
        return YOLO(yolo_config['model_path'])

    def _cv2_img_to_pil_img(self, cv2_img):
        """OpenCV图像转PIL图像"""
        return Image.fromarray(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB))

    def _pil_img_to_cv2_img(self, pil_img):
        """PIL图像转OpenCV图像"""
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    def _draw_chinese_text(self, cv2_img, text, pos, font_size=12, color=(0, 0, 255)):
        """绘制中文文本"""
        pil_img = self._cv2_img_to_pil_img(cv2_img)
        draw = ImageDraw.Draw(pil_img)


        draw.text(pos, text, font=self.font, fill=(color[2], color[1], color[0]))
        return self._pil_img_to_cv2_img(pil_img)

    def _load_font(self):
        """按 `yolo_detector` 的方式加载可用中文/通用字体，返回 (font, has_zh_font)"""
        candidate_paths = [
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/arphic/ukai.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "./simhei.ttf",
        ]
        for p in candidate_paths:
            try:
                if os.path.exists(p):
                    f = ImageFont.truetype(p, size=14)
                    # 判断是否为中文字体的简单启发式：文件名或路径包含 CJK 或 ukai 或 simhei
                    has_zh = any(x in p.lower() for x in ["cjk", "ukai", "simhei", "noto"])
                    return f, has_zh
            except Exception:
                continue
        try:
            return ImageFont.load_default(), False
        except Exception:
            return None, False

    def detect_sign(self, save_result_prefix, color_img):
        """使用YOLO检测并裁剪标牌"""
        yolo_config = self.config['model']['yolo']
        if color_img is None:
            raise ValueError(f"ocr_detector图像读取失败")

        # 执行YOLO检测
        results = self.yolo_model.predict(
            source=color_img,
            conf=yolo_config['conf_threshold'],
            iou=yolo_config['iou_threshold'],
            imgsz=yolo_config['imgsz'],
            device=self.device
        )

        # 处理检测结果
        annotator = Annotator(color_img.copy())
        cropped_list = []

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].int().tolist()
                conf = box.conf[0].item()
                cls = int(box.cls[0].item())
                cls_name = self.yolo_model.names[cls]

                # 绘制边界框
                annotator.box_label(
                    (x1, y1, x2, y2),
                    f"{cls_name} {conf:.2f}",
                    color=(0, 255, 0)
                )

                # 裁剪标牌区域
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(color_img.shape[1], x2), min(color_img.shape[0], y2)
                w, h = int(x2 - x1), int(y2 - y1)
                bbox = [x1, y1, w, h]
                if w <= 0 or h <= 0:
                    continue
                cropped = color_img[y1:y2, x1:x2]
                cropped_list.append((cropped, conf, bbox))

        # 保存可视化结果
        print("save_result_prefix:", save_result_prefix)
        cv2.imwrite(save_result_prefix + "_roiResult.png", annotator.result())

        # 保存最佳裁剪结果并返回裁剪图、置信度与bbox
        if not cropped_list:
            return None, None, None

        cropped_list.sort(key=lambda x: x[1], reverse=True)
        best_cropped, best_conf, best_bbox = cropped_list[0]
        cv2.imwrite(save_result_prefix + "_crop.png", best_cropped)

        return best_cropped, best_conf, best_bbox

    def _select_ocr_variant(self, candidates: Dict[str, Tuple[List[Any], List[_OCRItem]]]) -> str:
        def metrics(items: List[_OCRItem]) -> Tuple[float, float]:
            quality = sum(item.score ** 2 * max(len(item.text.strip()), 1) for item in items)
            mean_score = sum(item.score for item in items) / len(items) if items else 0.0
            return quality, mean_score

        qualities = {name: metrics(items)[0] for name, (_, items) in candidates.items()}
        best_quality = max(qualities.values())
        minimum = best_quality * (1.0 - float(self.preprocess_config['auto_quality_tie_ratio']))
        eligible = [name for name, quality in qualities.items() if quality >= minimum]
        preference = {'metal': 2, 'gray': 1, 'none': 0}
        return max(eligible, key=lambda name: (metrics(candidates[name][1])[1], qualities[name], preference[name]))

    def _get_line_recognizer(self):
        if self._line_recognizer is not None:
            return self._line_recognizer
        pipeline = getattr(getattr(self.ocr_model, 'paddlex_pipeline', None), '_pipeline', None)
        recognizer = getattr(pipeline, 'text_rec_model', None)
        if recognizer is None:
            recognizer = getattr(self.ocr_model, 'text_rec_model', None)
        if recognizer is not None and hasattr(recognizer, 'predict'):
            self._line_recognizer = recognizer
            return recognizer
        try:
            from paddleocr import TextRecognition
            ocr_config = self.config['model']['ocr']
            self._line_recognizer = TextRecognition(
                model_name='PP-OCRv5_server_rec',
                model_dir=ocr_config['text_recognition_model_dir'],
            )
        except (ImportError, KeyError, TypeError, ValueError, OSError) as exc:
            if self.config.get('debug'):
                print(f'无法初始化关键行纯识别模型: {exc}')
            return None
        return self._line_recognizer

    def _build_key_line_candidates(self, image: np.ndarray, items: List[_OCRItem], variant: str) -> List[_KeyLineCandidate]:
        if image.size == 0:
            return []
        _, width = image.shape[:2]
        inner_left = width * float(self.key_line_config.get('right_margin_ratio', 0.12))
        inner_right = width * (1.0 - float(self.key_line_config.get('right_margin_ratio', 0.12)))
        candidates = []
        seen_kinds = set()
        anchors = [(item, _classify_key_line(item.text)) for item in items]
        anchors = [(item, kind) for item, kind in anchors if kind is not None]
        for anchor, kind in anchors:
            if kind in seen_kinds:
                continue
            polygon = anchor.box.astype(np.float32)
            direction = ((polygon[1] - polygon[0]) + (polygon[2] - polygon[3])) / 2.0
            length = float(np.linalg.norm(direction))
            if length < 1.0:
                continue
            direction /= length
            if direction[0] < 0:
                direction *= -1.0
            normal = np.asarray([-direction[1], direction[0]], dtype=np.float32)
            anchor_s, anchor_t = polygon @ direction, polygon @ normal
            anchor_height = max(float(np.ptp(anchor_t)), 1.0)
            aligned = []
            for item in items:
                item_s, item_t = item.box @ direction, item.box @ normal
                item_height = max(float(np.ptp(item_t)), 1.0)
                item_width = max(float(np.ptp(item_s)), 1.0)
                is_aligned = abs(float(np.mean(item_t)) - float(np.mean(anchor_t))) <= max(anchor_height, item_height) * float(self.key_line_config.get('alignment_threshold_ratio', 0.55))
                is_inside = inner_left <= item.center[0] <= inner_right
                is_key_fragment = _classify_key_line(item.text) == kind
                is_isolated_alnum = bool(re.fullmatch(r'[A-Za-z0-9]', item.text.strip())) and item_height <= anchor_height * 1.6
                if is_aligned and (item is anchor or is_inside) and (item_width >= item_height * 0.75 or is_key_fragment or is_isolated_alnum):
                    aligned.append((item, item_s, item_t))
            if not aligned:
                aligned = [(anchor, anchor_s, anchor_t)]
            all_s = np.concatenate([value[1] for value in aligned])
            all_t = np.concatenate([value[2] for value in aligned])
            row_height = max(float(np.ptp(all_t)), anchor_height, 1.0)
            left_padding = float(self.key_line_config.get('left_padding_ratio', 0.75))
            if kind == '型号' and '型' not in re.sub(r'[\s:：]', '', anchor.text):
                left_padding = max(left_padding, 2.0)
            min_s = max(float(np.min(all_s)) - row_height * left_padding, float(np.dot([inner_left, anchor.center[1]], direction)))
            max_s = max(float(np.max(all_s)), float(np.dot([inner_right, anchor.center[1]], direction)))
            min_t = float(np.min(all_t)) - row_height * float(self.key_line_config.get('vertical_padding_ratio', 0.35))
            max_t = float(np.max(all_t)) + row_height * float(self.key_line_config.get('vertical_padding_ratio', 0.35))
            if max_s <= min_s or max_t <= min_t:
                continue
            box = np.asarray([direction * min_s + normal * min_t, direction * max_s + normal * min_t, direction * max_s + normal * max_t, direction * min_s + normal * max_t], dtype=np.float32)
            crop_width, crop_height = max(int(round(max_s - min_s)), 2), max(int(round(max_t - min_t)), 2)
            destination = np.asarray([[0, 0], [crop_width - 1, 0], [crop_width - 1, crop_height - 1], [0, crop_height - 1]], dtype=np.float32)
            crop = cv2.warpPerspective(image, cv2.getPerspectiveTransform(box, destination), (crop_width, crop_height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            sorted_items = sorted(aligned, key=lambda value: float(np.mean(value[1])))
            candidates.append(_KeyLineCandidate(
                kind=kind,
                detected_text=''.join(value[0].text for value in sorted_items),
                detected_score=min(value[0].score for value in sorted_items),
                preprocess_variant=variant,
                box=box,
                crop=crop,
                detected_content_end_x=int(round(np.clip(float(np.max(all_s)) - min_s, 0, crop_width - 1))),
            ))
            seen_kinds.add(kind)
        return candidates

    def _recover_key_fields(self, variants: Dict[str, np.ndarray], ocr_candidates: Dict[str, Tuple[List[Any], List[_OCRItem]]]) -> Dict[str, str]:
        if not self.key_line_config.get('enabled', True):
            return {}
        recognizer = self._get_line_recognizer()
        if recognizer is None:
            return {}
        candidates = []
        for variant, (_, items) in ocr_candidates.items():
            candidates.extend(self._build_key_line_candidates(variants[variant], items, variant))
        if not candidates:
            return {}
        try:
            raw_results = list(recognizer.predict(input=[candidate.crop for candidate in candidates]))
        except Exception as exc:
            if self.config.get('debug'):
                print(f'关键行识别失败: {exc}')
            return {}
        if len(raw_results) != len(candidates):
            return {}
        minimum_score = float(self.key_line_config.get('min_score', 0.3))
        attempts = []
        for candidate, raw_result in zip(candidates, raw_results):
            data = _unwrap_result_mapping(raw_result)
            recognized_text = str(data.get('rec_text', '')).strip()
            recognized_score = float(data.get('rec_score', 0.0))
            values = [(recognized_text, recognized_score, 'extended'), (candidate.detected_text, candidate.detected_score, 'detected')]
            text, score, method = max(values, key=lambda value: (
                _parse_key_field(candidate.kind, value[0])['format_valid'],
                _parse_key_field(candidate.kind, value[0])['complete'],
                _parse_key_field(candidate.kind, value[0])['value'] is not None,
                value[2] == 'extended', value[1], len(value[0]),
            ))
            attempts.append({'candidate': candidate, 'field': _parse_key_field(candidate.kind, text), 'score': score, 'method': method})

        suffix_jobs = []
        for attempt in attempts:
            candidate, field = attempt['candidate'], attempt['field']
            if not self.key_line_config.get('suffix_recovery_enabled', True) or candidate.kind != '型号' or candidate.preprocess_variant not in {'none', 'gray'} or field.get('base_model') != 'YZ-JXH-J' or field['complete']:
                continue
            start_x = min(candidate.detected_content_end_x + max(int(round(candidate.crop.shape[0] * 0.03)), 2), candidate.crop.shape[1])
            if candidate.crop.shape[1] - start_x < max(int(candidate.crop.shape[0] * 0.55), 8):
                continue
            for local_variant, crop in _build_local_relief_variants(candidate.crop[:, start_x:]).items():
                suffix_jobs.append((candidate, local_variant, crop))
        if suffix_jobs:
            try:
                suffix_results = list(recognizer.predict(input=[job[2] for job in suffix_jobs]))
                evidence = {}
                for (candidate, local_variant, _), raw_result in zip(suffix_jobs, suffix_results):
                    data = _unwrap_result_mapping(raw_result)
                    suffix = _extract_yunda_suffix(str(data.get('rec_text', '')))
                    if suffix:
                        evidence.setdefault(suffix, []).append((candidate, local_variant, float(data.get('rec_score', 0.0))))
                for suffix, entries in evidence.items():
                    neutral = [score for _, variant, score in entries if variant == 'local_gray']
                    variants_seen = {variant for _, variant, _ in entries}
                    scores = sorted((score for _, _, score in entries), reverse=True)
                    accepted = (neutral and max(neutral) >= float(self.key_line_config.get('suffix_high_confidence', 0.82))) or ({'local_dark_relief', 'local_bright_relief'} <= variants_seen and len(scores) >= 2 and sum(scores[:2]) / 2 >= max(float(self.key_line_config.get('suffix_consensus_min_score', 0.55)), 0.70))
                    if accepted:
                        candidate, _, score = max(entries, key=lambda value: value[2])
                        attempts.append({'candidate': candidate, 'field': _parse_key_field('型号', f'型号:YZ-JXH-J{suffix}'), 'score': score, 'method': 'local_relief'})
            except Exception as exc:
                if self.config.get('debug'):
                    print(f'型号后缀恢复失败: {exc}')

        fields = {}
        for kind in ('型号', 'No.'):
            valid = [attempt for attempt in attempts if attempt['candidate'].kind == kind and attempt['score'] >= minimum_score and attempt['field']['value']]
            if valid:
                best = max(valid, key=lambda attempt: (attempt['field']['format_valid'], attempt['field']['complete'], attempt['method'] != 'local_relief', attempt['score']))
                fields[kind] = best['field']['value']
        return fields

    def _recover_no_code(self, image: np.ndarray, items: List[_OCRItem]) -> Optional[str]:
        if not self.key_line_config.get('enabled', True):
            return None
        recognizer = self._get_line_recognizer()
        if recognizer is None:
            return None
        anchors = [item for item in items if re.search(r'(?<![a-z])no\s*\.?', item.text, re.IGNORECASE)]
        if not anchors:
            return None
        height, width = image.shape[:2]
        padding_ratio = float(self.key_line_config.get('vertical_padding_ratio', 0.35))
        margin_ratio = float(self.key_line_config.get('right_margin_ratio', 0.12))
        crops = []
        for anchor in anchors:
            row_height = max(anchor.height, 1.0)
            y1 = max(0, int(round(anchor.center[1] - row_height * (0.5 + padding_ratio))))
            y2 = min(height, int(round(anchor.center[1] + row_height * (0.5 + padding_ratio))))
            x1 = max(0, int(round(anchor.box[:, 0].min() - row_height * 0.75)))
            x2 = min(width, int(round(width * (1.0 - margin_ratio))))
            if x2 > x1 and y2 > y1:
                crops.append(image[y1:y2, x1:x2])
        if not crops:
            return None
        try:
            raw_results = list(recognizer.predict(input=crops))
        except Exception as exc:
            if self.config.get('debug'):
                print(f'关键行识别失败: {exc}')
            return None
        for raw_result in raw_results:
            try:
                result = _unwrap_result_mapping(raw_result)
                if float(result.get('rec_score', 0.0)) < float(self.key_line_config.get('min_score', 0.3)):
                    continue
                code = self._extract_no_code([str(result.get('rec_text', ''))])
                if code:
                    return code
            except (TypeError, ValueError, KeyError):
                continue
        return None

    def recognize_text(self, save_result_prefix, crop_image):
        """识别图像中的文本"""

        if crop_image is None:
            raise ValueError("无法处理图像数据")
        self._last_no_code = None
        self._last_key_fields = {}

        variants = _build_preprocess_variants(crop_image, self.preprocess_config)
        candidates = {}
        errors = {}
        for name, image in variants.items():
            try:
                raw_results = _materialize_prediction(self.ocr_model.predict(image))
                candidates[name] = (raw_results, _parse_ocr_items(raw_results, float(self.preprocess_config['min_score'])))
            except Exception as exc:
                errors[name] = str(exc)
        if not candidates:
            raise RuntimeError(f'OCR识别失败: {errors}')
        selected_variant = self._select_ocr_variant(candidates)
        result, parsed_items = candidates[selected_variant]
        if not result:
            raise ValueError("OCR识别结果为空")

        # 提取识别结果
        result_dict = _unwrap_result_mapping(result[0])
        required_keys = ["dt_polys", "rec_texts", "rec_scores"]
        for key in required_keys:
            if key not in result_dict:
                raise KeyError(f"OCR结果缺少关键键: {key}")

        dt_polys = result_dict["dt_polys"]
        rec_texts = result_dict["rec_texts"]
        rec_scores = result_dict["rec_scores"]

        # 校验数据长度
        if not (len(dt_polys) == len(rec_texts) == len(rec_scores)):
            raise ValueError("OCR结果数据长度不匹配")
        
        img = variants[selected_variant].copy()
        
        # 绘制识别结果
        for idx in range(len(rec_texts)):
            text = rec_texts[idx]
            score = rec_scores[idx]
            poly = np.asarray(dt_polys[idx])

            if poly.shape[0] != 4:
                if self.config['debug']:
                    print(f"警告：第{idx}个文本框坐标异常，跳过")
                continue

            # 绘制文本框
            box = poly.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(img, [box], isClosed=True, color=(0, 255, 0), thickness=2)

            # 绘制文本
            text_pos = (box[0][0][0], box[0][0][1] - 10)
            if text_pos[1] < 10:
                text_pos = (box[0][0][0], box[0][0][1] + 20)

            img = self._draw_chinese_text(
                img,
                f"{text}({score:.2f})",
                text_pos,
                font_size=14,
                color=(0, 0, 255)
            )

        # 保存OCR可视化结果
        cv2.imwrite(save_result_prefix + "_ocr.png", img)
        self._last_key_fields = self._recover_key_fields(variants, candidates)
        self._last_no_code = self._last_key_fields.get('No.') or self._recover_no_code(
            variants[selected_variant], parsed_items
        )

        return {
            "texts": rec_texts,
            "scores": rec_scores,
            "boxes": dt_polys
        }

    def _extract_no_code(self, texts: List[str]) -> Optional[str]:
        """从OCR文本中提取 No 后面的一位数字加一位字母。"""
        def clean_token(token: str) -> str:
            return re.sub(r"[^0-9A-Za-z]", "", token)

        # 先尝试整段匹配
        for text in texts:
            if not text:
                continue
            # 支持 No123A、No 1A、No: 1A、No；1；A、No;1;A、No-1-A 等格式
            m = re.search(r"\bNo\.?[\s:；;,_\-]*([0-9])[\s:；;,_\-]*([A-Za-z])\b", text, flags=re.IGNORECASE)
            if m:
                return (m.group(1) + m.group(2)).upper()

            # 兼容紧接的纯数字+字母，如 No1A 或 no1a
            m2 = re.search(r"\bNo([0-9][A-Za-z])\b", text, flags=re.IGNORECASE)
            if m2:
                return m2.group(1).upper()

        # 文本可能被分成多个 OCR 结果，例如 ["No", "1", "A"] 或 ["No", ";", "1", ";", "A"]
        cleaned = [clean_token(t) for t in texts if t and clean_token(t)]
        for i, token in enumerate(cleaned):
            if token.lower() == "no":
                next_tokens = cleaned[i + 1 : i + 3]
                if len(next_tokens) >= 2 and next_tokens[0].isdigit() and len(next_tokens[1]) == 1 and next_tokens[1].isalpha():
                    return (next_tokens[0] + next_tokens[1]).upper()

        return None

    def _extract_template_no_code(self, point_name: Optional[str]) -> Optional[str]:
        """从项点名称中的 No. 字段提取模板要求的编号。"""
        if not point_name:
            return None
        match = re.search(
            r"No\.?[\s:：；;,_\-]*([0-9])[\s:：；;,_\-]*([A-Za-z])",
            str(point_name),
            flags=re.IGNORECASE,
        )
        return (match.group(1) + match.group(2)).upper() if match else None

    def infer(
        self,
        save_result_prefix: str,
        color_img: np.ndarray,
        depth_img: np.ndarray,
        train_type_name: Optional[str] = None,
        expected_point_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """完整处理流程：检测标牌 -> 识别文字"""
        cropped_sign, best_conf, bbox = self.detect_sign(save_result_prefix, color_img)
        if cropped_sign is None:
            return {
                "analyse_result": "error",
                "message": "ocr_detector未检测到标牌区域",
                "detections": []
            }

        ocr_results = self.recognize_text(save_result_prefix, cropped_sign)
        if not ocr_results["texts"]:
            return {
                "analyse_result": "error",
                "message": "未识别到文字",
                "detections": []
            }

        no_code = self._last_no_code or self._extract_no_code(ocr_results["texts"])
        if no_code is None:
            return {
                "analyse_result": "error",
                "message": "未识别到 No 后面的数字字母",
                "detections": []
            }

        expected_no_code = self._extract_template_no_code(expected_point_name)
        name_match = not expected_no_code or no_code == expected_no_code
        result_status = "正常" if name_match else "异常"
        message = (
            f"识别结果：No.{no_code}，"
            f"模板要求：No.{expected_no_code or '未提取'}，"
            f"检测结果：{result_status}"
        )
        return {
            "analyse_result": "ok" if name_match else "error",
            "message": message,
            "detections": [
                {
                    "abnormal_obj": not name_match,
                    "type_code": "ocr_no_code",
                    "type_name": no_code,
                    "confidence": 1.0,
                    "bbox": {"x": bbox[0], "y": bbox[1], "w": bbox[2], "h": bbox[3]},
                    "status": "正常" if name_match else "异常",
                }
            ]
        }

if __name__ == "__main__":
    from datetime import datetime

    config_path = resolve_path("$(find detect)/config/ocr_detector.yaml")
    image_dir = "/home/shansu/moma_ws/src/detect/tests/ocr"
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"/home/shansu/moma_ws/src/detect/output/ocr/{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    detector = OCRDetector(config_path)

    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    image_files = sorted([
        f for f in os.listdir(image_dir)
        if os.path.splitext(f)[1].lower() in valid_exts
    ])

    print(f"开始批量测试，共 {len(image_files)} 张图片")
    print(f"输出路径: {output_dir}")

    for idx, filename in enumerate(image_files, 1):
        image_path = os.path.join(image_dir, filename)
        color_img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if color_img is None:
            print(f"[{idx}/{len(image_files)}] 读取图像失败: {filename}")
            continue

        prefix_name = os.path.splitext(filename)[0]
        save_result_prefix = os.path.join(output_dir, prefix_name)

        print(f"[{idx}/{len(image_files)}] 正在处理: {filename}")
        try:
            result = detector.infer(save_result_prefix, color_img, None)
            print(f"  -> 结果: {result.get('message', result)}")
        except Exception as e:
            print(f"  -> 处理异常: {e}")
