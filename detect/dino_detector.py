import os
import re
from collections import defaultdict

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import networkx as nx  # 用于绘制无向图
from sklearn.metrics.pairwise import cosine_similarity
import yaml
# from transformers import AutoImageProcessor, AutoModel
from modelscope import AutoImageProcessor, AutoModel

try:
    from .path_utils import load_yaml_config, resolve_ros_substitutions
except ImportError:
    from path_utils import load_yaml_config, resolve_ros_substitutions

# 解决字体问题：使用系统更可能存在的中文字体
plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题


class DINODetector:
    """
    使用 DINOv2 / DINOv3 提取图像全局特征，进行相似图检索 + 类别判断。
    IMAGE_DIR 结构建议为：
        IMAGE_DIR/
            classA/
                a1.jpg
                a2.jpg
            classB/
                b1.jpg
                ...

    feature_type:
        - "pooler": 使用 outputs.pooler_output
        - "cls": 使用最后一层 CLS token
        - "mean_patch": 对 patch token 做平均池化（默认，推荐用于检索）
    """

    def __init__(self, model_path, image_dir, feature_type="mean_patch", config_path=None):
        if config_path is not None and os.path.exists(config_path):
            cfg = load_yaml_config(config_path)
            model_path = model_path or cfg.get("model_path")
            image_dir = image_dir or cfg.get("image_dir")
            feature_type = feature_type or cfg.get("feature_type", "mean_patch")

        self.model_path = self._resolve_path(resolve_ros_substitutions(model_path))
        self.image_dir = self._resolve_path(resolve_ros_substitutions(image_dir))
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        assert feature_type in {"pooler", "cls", "mean_patch"}
        self.feature_type = feature_type

        # 加载模型和处理器
        self.processor = AutoImageProcessor.from_pretrained(self.model_path)
        self.model = AutoModel.from_pretrained(self.model_path).to(self.device)
        self.model.eval()
        self.image_dir_root = self._resolve_path(image_dir)
        self.train_type_name = None

        # 存储图片路径、名称、类别和特征
        self.image_paths = []
        self.image_names = []
        self.image_labels = []  # 类别名（由子文件夹推断）
        self.features = None

        # 查询图相关信息（用于可视化）
        self.query_feat = None
        self.query_name = None
        self.query_image = None

    def set_train_type(self, train_type_name):
        """切换车型图片库；车型变化后下次推理时重新构建特征库。"""
        train_type_name = str(train_type_name or "").strip()
        if not train_type_name:
            raise ValueError("车型不能为空")
        image_dir = os.path.join(self.image_dir_root, train_type_name)
        if not os.path.isdir(image_dir):
            raise FileNotFoundError(f"未找到车型图片库: {image_dir}")
        if train_type_name != self.train_type_name:
            self.train_type_name = train_type_name
            self.image_dir = image_dir
            self.features = None

    @staticmethod
    def _resolve_path(path):
        if path is None:
            return None
        if os.path.isabs(path):
            return path
        if os.path.exists(path):
            return os.path.abspath(path)

        candidates = [
            os.path.join(os.getcwd(), path),
            os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")), path),
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                return os.path.abspath(candidate)
        return path

    @staticmethod
    def _safe_dir_name(name):
        safe_name = str(name).strip()
        safe_name = safe_name.replace(os.sep, "_")
        safe_name = safe_name.replace("/", "_")
        safe_name = safe_name.replace("\\", "_")
        safe_name = safe_name.replace(" ", "_")
        return safe_name or "class"

    @staticmethod
    def _extract_expected_type_name(point_name):
        """从“X位-外端盖名称”中提取外端盖名称。"""
        value = str(point_name or "").strip()
        match = re.match(r"^\s*[^-—–－]+位\s*[-—–－]\s*(.+?)\s*$", value)
        return match.group(1).strip() if match else value

    def _infer_label(self, root, filename):
        """保留兼容接口；实际分类不再依赖文件名或目录名。"""
        rel_dir = os.path.relpath(root, self.image_dir)
        if rel_dir != ".":
            return rel_dir.split(os.sep)[0]
        return os.path.basename(self.image_dir)

    # ------------------ 特征提取 ------------------ #
    def _extract_single_feature(self, image: Image.Image) -> np.ndarray:
        """
        提取单张图片的全局特征，返回 numpy 数组 [1, D]。
        """
        with torch.inference_mode():
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)

            if (
                self.feature_type == "pooler"
                and hasattr(outputs, "pooler_output")
                and outputs.pooler_output is not None
            ):
                # 直接使用 pooler_output
                feat = outputs.pooler_output  # [B, D]
            else:
                # 使用最后一层 hidden state
                hidden = outputs.last_hidden_state  # [B, T, C]
                # CLS 在第一个 token，patch 紧随其后，register token 在最后若干个位置
                n_tokens = hidden.shape[1]
                n_reg = getattr(self.model.config, "num_register_tokens", 0)

                if self.feature_type == "cls":
                    feat = hidden[:, 0, :]  # [B, C]
                else:
                    # mean_patch: 只对 patch token 做平均
                    start = 1  # 跳过 CLS
                    end = n_tokens - n_reg if n_reg > 0 else n_tokens
                    patch_tokens = hidden[:, start:end, :]  # [B, N_patch, C]
                    feat = patch_tokens.mean(dim=1)  # [B, C]

            # 归一化有利于余弦相似度检索
            feat = F.normalize(feat, dim=-1)
            return feat.cpu().numpy()  # [1, D]

    # ------------------ 图片库构建 ------------------ #
    def build_image_library(self):
        """
        递归遍历 image_dir，收集所有图片路径、类别，并一次性提取特征。
        """
        print(f"正在构建图片库: {self.image_dir}")
        img_ext = (".png", ".jpg", ".jpeg", ".bmp", ".gif")

        image_records = []
        for root, _, files in os.walk(self.image_dir):
            for f in files:
                if not f.lower().endswith(img_ext):
                    continue
                path = os.path.join(root, f)
                label = self._infer_label(root, f)
                image_records.append((path, os.path.basename(path), label))

        if not image_records:
            raise RuntimeError(f"在 {self.image_dir} 下未找到任何图片。")

        # 批量提取特征
        features_list = []
        valid_records = []
        for img_path, image_name, label in image_records:
            try:
                img = Image.open(img_path).convert("RGB")
                feat = self._extract_single_feature(img)
                features_list.append(feat)
                valid_records.append((img_path, image_name, label))
            except Exception as e:
                print(f"⚠️ 跳过图片 {img_path}：{e}")

        if not features_list:
            raise RuntimeError("图片库中没有成功提取特征的图片。")

        self.image_paths, self.image_names, self.image_labels = map(
            list, zip(*valid_records)
        )
        self.features = np.vstack(features_list)
        print(f"图片库构建完成：共 {len(self.image_paths)} 张图片，"
              f"{len(set(self.image_labels))} 个类别。")

    # ------------------ 输入预处理 ------------------ #
    def _prepare_image(self, image_input):
        """支持从路径、PIL 图像或 numpy 数组中读取查询图像。"""
        if isinstance(image_input, Image.Image):
            return image_input.convert("RGB")

        if isinstance(image_input, (str, os.PathLike)):
            return Image.open(image_input).convert("RGB")

        if isinstance(image_input, np.ndarray):
            arr = np.asarray(image_input)
            if arr.ndim == 2:
                arr = np.stack([arr] * 3, axis=-1)
            elif arr.ndim == 3 and arr.shape[2] == 4:
                arr = arr[..., :3]
            elif arr.ndim == 3 and arr.shape[2] == 3:
                # ROS/OpenCV 输入通常是 BGR，转换为 RGB 以便与模型一致
                arr = arr[:, :, ::-1]
            return Image.fromarray(arr.astype(np.uint8))

        raise TypeError(f"不支持的图像输入类型: {type(image_input)}")

    def infer(
        self,
        save_path_prefix: str,
        color_img,
        depth_img=None,
        train_type_name=None,
        expected_point_name=None,
    ):
        """对输入图像执行 DINO 检索并返回与 detect.py 兼容的结果结构。"""
        try:
            if color_img is None:
                return {
                    "analyse_result": "error",
                    "message": "输入图像为空",
                    "detections": [],
                }

            if train_type_name:
                self.set_train_type(train_type_name)
            elif self.train_type_name is None:
                raise ValueError("DINO 推理必须提供车型")

            if self.features is None:
                self.build_image_library()

            # 复用已有检索/聚类逻辑
            similar_images, ranked_classes = self.find_similar(
                color_img,
                top_n=5,
                class_agg="max",
                return_class_scores=True,
            )

            if not ranked_classes:
                return {
                    "analyse_result": "error",
                    "message": "DINO 检索未能生成类别结果",
                    "detections": [],
                }

            top_classes = [
                {
                    "type_name": label,
                    "confidence": float(score),
                    "type_code": f"dino_{label}",
                }
                for label, score in ranked_classes
            ]

            best_label, best_score = ranked_classes[0]
            template_point_name = str(expected_point_name or "").strip()
            expected_type_name = self._extract_expected_type_name(template_point_name)
            name_match = bool(expected_type_name) and best_label == expected_type_name
            result_status = "正常" if name_match else "异常"
            detections = []
            for label, score in ranked_classes:
                label_match = label == expected_type_name
                detections.append(
                    {
                        "abnormal_obj": not label_match,
                        "type_code": f"dino_{label}",
                        "type_name": label,
                        "confidence": float(score),
                        "bbox": {"x": 0, "y": 0, "w": 0, "h": 0},
                        "status": "正常" if label_match else "异常",
                        "class_scores": top_classes,
                    }
                )

            message = (
                f"识别结果：{best_label}，"
                f"模板要求：{expected_type_name}，检测结果：{result_status}"
            )

            outcome_root = save_path_prefix
            if not os.path.isabs(outcome_root):
                outcome_root = os.path.abspath(outcome_root)
            os.makedirs(outcome_root, exist_ok=True)

            for label, score in ranked_classes:
                class_dir = os.path.join(outcome_root, self._safe_dir_name(label))
                os.makedirs(class_dir, exist_ok=True)
                with open(os.path.join(class_dir, "result.txt"), "w", encoding="utf-8") as f:
                    f.write(f"label={label}\nscore={score:.6f}\n")

            source_name = self.query_name or "query.jpg"
            source_stem, source_ext = os.path.splitext(source_name)
            source_ext = source_ext or ".jpg"
            class_dir = os.path.join(outcome_root, self._safe_dir_name(best_label))
            os.makedirs(class_dir, exist_ok=True)
            target_path = os.path.join(class_dir, f"{source_stem}{source_ext}")
            try:
                Image.fromarray(np.array(self.query_image)).save(target_path)
            except Exception:
                try:
                    self.query_image.save(target_path)
                except Exception:
                    pass

            return {
                "analyse_result": "ok" if name_match else "error",
                "message": message,
                "detections": detections,
            }
        except Exception as exc:
            return {
                "analyse_result": "error",
                "message": f"DINO 推理失败: {exc}",
                "detections": [],
            }

    # ------------------ 相似检索 ------------------ #
    def find_similar(
        self,
        query_image_path,
        top_n: int = 5,
        class_agg: str = "max",
        return_class_scores: bool = True,
    ):
        """
        对 query_img 做检索，返回 Top-N 相似图片，并根据相似度聚合得到类别得分。

        query_image_path 可以是图片路径，也可以是 PIL.Image.Image / numpy.ndarray。

        class_agg:
            - "max": 每个类别取该类别中「最高相似度」作为得分（更像最近邻）
            - "mean": 每个类别取该类别中「平均相似度」作为得分（更稳）
        """
        if self.features is None:
            self.build_image_library()

        # 支持路径或内存图像
        if isinstance(query_image_path, (str, os.PathLike)):
            self.query_image = Image.open(query_image_path).convert("RGB")
            self.query_name = os.path.basename(query_image_path)
        else:
            self.query_image = self._prepare_image(query_image_path)
            self.query_name = getattr(query_image_path, "filename", None) or "query"

        self.query_feat = self._extract_single_feature(self.query_image)

        # 计算余弦相似度（值越大越相似）
        similarities = cosine_similarity(self.query_feat, self.features).flatten()  # [N]

        # Top-N 相似图片（排除自身路径）
        indices_sorted = similarities.argsort()[::-1]
        top_indices = []
        for idx in indices_sorted:
            if isinstance(query_image_path, (str, os.PathLike)) and self.image_paths[idx] == query_image_path:
                continue
            top_indices.append(idx)
            if len(top_indices) >= top_n:
                break

        similar_images = [
            (
                self.image_paths[i],      # img_path
                self.image_names[i],      # img_name
                self.image_labels[i],     # img_label
                float(similarities[i]),   # sim
            )
            for i in top_indices
        ]

        if not return_class_scores:
            return similar_images

        # 聚合类别得分
        class_sims = defaultdict(list)
        for s, lab in zip(similarities, self.image_labels):
            class_sims[lab].append(float(s))

        if class_agg == "mean":
            class_scores = {
                lab: float(np.mean(vals)) for lab, vals in class_sims.items()
            }
        else:  # "max"
            class_scores = {
                lab: float(np.max(vals)) for lab, vals in class_sims.items()
            }

        # 从大到小排序，便于打印 / 可视化
        class_scores_sorted = sorted(
            class_scores.items(), key=lambda x: x[1], reverse=True
        )

        return similar_images, class_scores_sorted

    # ------------------ 可视化 ------------------ #
    def visualize_image_grid(
        self,
        similar_images,
        class_scores=None,
        top_k_classes: int = 3,
    ):
        """
        网格展示查询图 + Top-N 相似图。
        similar_images: [(img_path, img_name, img_label, sim), ...]
        """
        n = len(similar_images) + 1
        fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))

        # 显示查询图
        if self.query_image is None:
            raise RuntimeError("请先调用 find_similar 再进行可视化。")

        axes[0].imshow(self.query_image)
        axes[0].set_title(f"查询图片\n{self.query_name}", fontsize=10)
        axes[0].axis("off")

        # 显示相似图
        for i, (img_path, img_name, img_label, sim) in enumerate(similar_images, 1):
            img = Image.open(img_path).convert("RGB")
            axes[i].imshow(img)
            axes[i].set_title(
                f"{img_name}\n类: {img_label}\ncos={sim:.3f}",
                fontsize=9,
            )
            axes[i].axis("off")

        # 如果有类别得分，在整体标题中标出预测类别
        if class_scores is not None and len(class_scores) > 0:
            pred_label, pred_score = class_scores[0]
            sup_title = f"预测类别：{pred_label}（得分 {pred_score:.3f}）"
        else:
            sup_title = "查询图片与最相似的图片对比"

        plt.suptitle(sup_title, y=1.02, fontsize=12)
        plt.tight_layout()
        plt.show()

        # 可选：画一个类别得分条形图
        if class_scores is not None:
            self.visualize_class_scores(class_scores, top_k=top_k_classes)

    def visualize_class_scores(self, class_scores, top_k=5):
        """
        类别得分条形图。class_scores: [(label, score), ...] 已按得分降序排好。
        """
        top_k = min(top_k, len(class_scores))
        labels = [lab for lab, _ in class_scores[:top_k]]
        scores = [s for _, s in class_scores[:top_k]]

        plt.figure(figsize=(6, 4))
        plt.bar(range(top_k), scores)
        plt.xticks(range(top_k), labels, rotation=30)
        plt.ylabel("聚合相似度（cos）")
        plt.title("Top 类别相似度（用于类别判断）")
        plt.tight_layout()
        plt.show()

    def visualize_similarity_graph(self, similar_images):
        """
        用无向图展示查询图与相似图的相似度关系（距离越近越相似）。
        similar_images: [(img_path, img_name, img_label, sim), ...]
        """
        if self.query_name is None:
            raise RuntimeError("请先调用 find_similar 再进行可视化。")

        G = nx.Graph()

        query_node = f"{self.query_name}\n[query]"
        G.add_node(query_node, type="query")

        sim_nodes = []
        for _, img_name, img_label, _ in similar_images:
            node_id = f"{img_name}\n[{img_label}]"
            sim_nodes.append(node_id)
            G.add_node(node_id, type="similar")

        for (_, img_name, img_label, sim) in similar_images:
            node_id = f"{img_name}\n[{img_label}]"
            G.add_edge(query_node, node_id, weight=sim)

        pos = nx.spring_layout(G, weight="weight", seed=42)

        # 绘制节点
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=[query_node],
            node_color="lightgreen",
            node_shape="o",
            node_size=800,
        )
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=sim_nodes,
            node_color="lightblue",
            node_shape="o",
            node_size=800,
        )

        # 绘制边（粗细与相似度正相关）
        edges = G.edges()
        weights = [G[u][v]["weight"] * 5 for u, v in edges]
        nx.draw_networkx_edges(G, pos, edgelist=edges, width=weights, alpha=0.7)

        nx.draw_networkx_labels(G, pos, font_size=8)

        edge_labels = {(u, v): f"{G[u][v]['weight']:.2f}" for u, v in edges}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)

        plt.title("查询图片与相似图片的相似度图（距离越近/线越粗 = 越相似）")
        plt.axis("off")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # ===== 配置路径（请按需修改） =====
    MODEL_PATH = "models/facebook/dinov3-vits16"  # DINO 模型路径
    IMAGE_DIR = "images/batch0"                   # 图片库（包含子文件夹）
    QUERY_IMAGE = "query4.jpg"                       # 查询图片路径
    TOP_N = 5                                     # Top-N 相似图片
    FEATURE_TYPE = "pooler"                   # "pooler" | "cls" | "mean_patch"

    searcher = DINODetector(
        model_path=MODEL_PATH,
        image_dir=IMAGE_DIR,
        feature_type=FEATURE_TYPE,
    )

    similar, class_scores = searcher.find_similar(
        QUERY_IMAGE,
        top_n=TOP_N,
        class_agg="max",          # 或 "mean"
        return_class_scores=True,
    )

    # 打印结果
    print(f"\n与 {QUERY_IMAGE} 最相似的 {TOP_N} 张图片：")
    for i, (path, name, label, sim) in enumerate(similar, 1):
        print(f"{i}. [{label}] {name} （相似度：{sim:.4f}） -> {path}")

    if class_scores:
        print("\n类别得分（从高到低）：")
        for label, score in class_scores:
            print(f"- {label}: {score:.4f}")
        pred_label, pred_score = class_scores[0]
        print(f"\n预测类别：{pred_label}（得分 {pred_score:.4f}）")

    # 可视化：网格 + 类别条形图 + 可选图结构
    searcher.visualize_image_grid(similar, class_scores, top_k_classes=5)
    # 如需，再画相似度图：
    searcher.visualize_similarity_graph(similar)
