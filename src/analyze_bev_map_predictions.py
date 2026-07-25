#!/usr/bin/env python3
"""Analyze BEV map predictions and explain the IoU metrics.

This script reads the pickle produced by ``bevfusion-main/tools/test.py --out``.
It recomputes the same TP/FP/FN/IoU sweep as ``NuScenesDataset.evaluate_map``
and writes human-readable artifacts for debugging:

* summary_metrics.json: metric keys matching evaluate_map.
* class_thresholds.csv: global TP/FP/FN/IoU by class and threshold.
* sample_thresholds.csv: per-sample TP/FP/FN/IoU by class and threshold.
* analysis.md: short explanation, formula, and high-level tables.
* visualizations/*.png: GT, probability, binary prediction, and TP/FP/FN overlay.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


DEFAULT_CLASSES = ("drivable_area", "ped_crossing", "divider")
DEFAULT_THRESHOLDS = (0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65)


def load_pickle(path: Path) -> Any:
    """Load a pickle using mmcv when available, then fall back to pickle."""
    try:
        import mmcv  # type: ignore

        return mmcv.load(str(path))
    except Exception:
        with path.open("rb") as handle:
            return pickle.load(handle)


def to_numpy(value: Any) -> np.ndarray:
    """Convert torch tensors, numpy arrays, or array-like values to numpy."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def read_infos(path: Optional[Path]) -> Optional[List[dict]]:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(path)
    data = load_pickle(path)
    if isinstance(data, dict) and "infos" in data:
        return list(data["infos"])
    if isinstance(data, list):
        return data
    raise ValueError(f"{path} must be a list or a dict containing 'infos'")


def sample_identity(index: int, infos: Optional[Sequence[dict]]) -> Tuple[str, Optional[int]]:
    if infos is None or index >= len(infos):
        return f"sample_{index:06d}", None
    info = infos[index]
    token = str(info.get("token", f"sample_{index:06d}"))
    timestamp = info.get("timestamp")
    return token, int(timestamp) if timestamp is not None else None


def format_float(value: float, digits: int = 8) -> str:
    if math.isnan(value):
        return "nan"
    return f"{value:.{digits}f}"


def percentile_from_hist(hist: np.ndarray, bin_edges: np.ndarray, percentile: float) -> float:
    total = int(hist.sum())
    if total <= 0:
        return float("nan")
    target = total * percentile / 100.0
    cumulative = np.cumsum(hist)
    index = int(np.searchsorted(cumulative, target, side="left"))
    index = min(index, len(bin_edges) - 2)
    return float((bin_edges[index] + bin_edges[index + 1]) / 2.0)


def compute_iou(tp: np.ndarray, fp: np.ndarray, fn: np.ndarray) -> np.ndarray:
    denominator = tp + fp + fn
    return np.divide(
        tp,
        denominator + 1e-7,
        out=np.zeros_like(tp, dtype=np.float64),
        where=denominator >= 0,
    )


def validate_shapes(prediction: np.ndarray, label: np.ndarray, classes: Sequence[str]) -> None:
    if prediction.shape != label.shape:
        raise ValueError(f"prediction shape {prediction.shape} != label shape {label.shape}")
    if prediction.ndim != 3:
        raise ValueError(f"expected masks with shape [C,H,W], got {prediction.shape}")
    if prediction.shape[0] != len(classes):
        raise ValueError(
            f"expected {len(classes)} classes, got prediction shape {prediction.shape}"
        )


def parse_indices(text: Optional[str]) -> Optional[List[int]]:
    if not text:
        return None
    indices: List[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        indices.append(int(part))
    return indices


def choose_visualization_indices(
    sample_scores: Sequence[dict],
    requested: Optional[Sequence[int]],
    max_viz: int,
) -> List[int]:
    if requested:
        return list(dict.fromkeys(int(i) for i in requested))
    if max_viz <= 0 or not sample_scores:
        return []

    by_score = sorted(sample_scores, key=lambda row: row["mean_best_iou"])
    chosen: List[int] = []

    def add(index: int) -> None:
        if index not in chosen and 0 <= index < len(sample_scores):
            chosen.append(index)

    add(0)
    add(len(sample_scores) // 2)
    add(len(sample_scores) - 1)

    half = max(1, max_viz // 2)
    for row in by_score[:half]:
        add(row["index"])
        if len(chosen) >= max_viz:
            return chosen
    for row in reversed(by_score[-half:]):
        add(row["index"])
        if len(chosen) >= max_viz:
            return chosen
    return chosen[:max_viz]


def probability_heatmap(prob: np.ndarray) -> np.ndarray:
    prob = np.nan_to_num(prob.astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    prob = np.clip(prob, 0.0, 1.0)
    rgb = np.zeros((*prob.shape, 3), dtype=np.uint8)
    rgb[..., 0] = np.clip(255.0 * prob, 0, 255).astype(np.uint8)
    rgb[..., 1] = np.clip(255.0 * np.sqrt(prob), 0, 255).astype(np.uint8)
    rgb[..., 2] = np.clip(255.0 * (1.0 - prob), 0, 255).astype(np.uint8)
    return rgb


def bool_panel(mask: np.ndarray, color: Tuple[int, int, int]) -> np.ndarray:
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    rgb[:] = (35, 35, 35)
    rgb[mask.astype(bool)] = color
    return rgb


def overlay_panel(pred_binary: np.ndarray, label: np.ndarray) -> np.ndarray:
    pred_binary = pred_binary.astype(bool)
    label = label.astype(bool)
    tp = pred_binary & label
    fp = pred_binary & ~label
    fn = ~pred_binary & label

    rgb = np.zeros((*label.shape, 3), dtype=np.uint8)
    rgb[:] = (45, 45, 45)
    rgb[tp] = (40, 190, 90)
    rgb[fp] = (230, 65, 65)
    rgb[fn] = (65, 130, 230)
    return rgb


def draw_labeled_tiles(
    tiles: Sequence[Tuple[str, np.ndarray]],
    output_path: Path,
    scale: int = 3,
) -> None:
    from PIL import Image, ImageDraw

    if not tiles:
        return
    label_height = 22
    gap = 8
    tile_h, tile_w = tiles[0][1].shape[:2]
    scaled_w = tile_w * scale
    scaled_h = tile_h * scale
    canvas_w = len(tiles) * scaled_w + (len(tiles) - 1) * gap
    canvas_h = scaled_h + label_height
    canvas = Image.new("RGB", (canvas_w, canvas_h), (20, 20, 20))
    draw = ImageDraw.Draw(canvas)

    x = 0
    for label, array in tiles:
        image = Image.fromarray(array).resize((scaled_w, scaled_h), Image.NEAREST)
        canvas.paste(image, (x, 0))
        draw.text((x + 4, scaled_h + 4), label, fill=(235, 235, 235))
        x += scaled_w + gap

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def safe_filename(text: str) -> str:
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_")
    return "".join(ch if ch in allowed else "_" for ch in text)


def write_selected_tensor_npz(
    outputs: Sequence[dict],
    output_dir: Path,
    classes: Sequence[str],
    thresholds: np.ndarray,
    class_best_threshold_indices: Sequence[int],
    sample_indices: Sequence[int],
    infos: Optional[Sequence[dict]],
) -> List[str]:
    written: List[str] = []
    tensor_dir = output_dir / "selected_tensors"
    tensor_dir.mkdir(parents=True, exist_ok=True)
    best_thresholds = np.asarray(
        [thresholds[index] for index in class_best_threshold_indices],
        dtype=np.float32,
    )

    for sample_index in sample_indices:
        if sample_index < 0 or sample_index >= len(outputs):
            continue
        token, timestamp = sample_identity(sample_index, infos)
        prediction = to_numpy(outputs[sample_index]["masks_bev"]).astype(np.float32)
        label = to_numpy(outputs[sample_index]["gt_masks_bev"]).astype(bool)
        validate_shapes(prediction, label, classes)

        pred_binary = prediction >= best_thresholds[:, None, None]
        match_codes = np.zeros(label.shape, dtype=np.uint8)
        match_codes[pred_binary & label] = 1  # TP
        match_codes[pred_binary & ~label] = 2  # FP
        match_codes[~pred_binary & label] = 3  # FN

        path = tensor_dir / f"{sample_index:06d}_{safe_filename(token)}_tensors.npz"
        np.savez_compressed(
            path,
            masks_bev=prediction,
            gt_masks_bev=label.astype(np.uint8),
            pred_binary_best=pred_binary.astype(np.uint8),
            match_codes_best=match_codes,
            best_thresholds=best_thresholds,
            thresholds=thresholds.astype(np.float32),
            classes=np.asarray(classes),
            sample_index=np.asarray(sample_index, dtype=np.int64),
            token=np.asarray(token),
            timestamp=np.asarray(-1 if timestamp is None else timestamp, dtype=np.int64),
        )
        written.append(str(path.relative_to(output_dir)))
    return written


def write_visualizations(
    outputs: Sequence[dict],
    output_dir: Path,
    classes: Sequence[str],
    thresholds: np.ndarray,
    class_best_threshold_indices: Sequence[int],
    sample_indices: Sequence[int],
    infos: Optional[Sequence[dict]],
    viz_threshold: Optional[float],
    scale: int,
) -> List[str]:
    try:
        import PIL  # noqa: F401
    except Exception as exc:
        return [f"Skipped image generation because Pillow is unavailable: {exc}"]

    written: List[str] = []
    viz_dir = output_dir / "visualizations"
    for sample_index in sample_indices:
        if sample_index < 0 or sample_index >= len(outputs):
            continue
        token, _ = sample_identity(sample_index, infos)
        result = outputs[sample_index]
        prediction = to_numpy(result["masks_bev"]).astype(np.float32)
        label = to_numpy(result["gt_masks_bev"]).astype(bool)
        validate_shapes(prediction, label, classes)

        for class_index, class_name in enumerate(classes):
            threshold = (
                float(viz_threshold)
                if viz_threshold is not None
                else float(thresholds[class_best_threshold_indices[class_index]])
            )
            pred_binary = prediction[class_index] >= threshold
            tiles = [
                ("GT", bool_panel(label[class_index], (240, 240, 240))),
                ("Pred prob", probability_heatmap(prediction[class_index])),
                (f"Pred >= {threshold:.2f}", bool_panel(pred_binary, (245, 190, 70))),
                ("TP green FP red FN blue", overlay_panel(pred_binary, label[class_index])),
            ]
            path = viz_dir / f"{sample_index:06d}_{safe_filename(token)}_{class_name}.png"
            draw_labeled_tiles(tiles, path, scale=scale)
            written.append(str(path.relative_to(output_dir)))
    return written


def write_csv(path: Path, rows: Iterable[dict], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_markdown(
    path: Path,
    predictions_path: Path,
    info_path: Optional[Path],
    classes: Sequence[str],
    thresholds: np.ndarray,
    sample_count: int,
    mask_shape: Sequence[int],
    metrics: Dict[str, float],
    class_rows: Sequence[dict],
    artifacts: Sequence[str],
    image_files: Sequence[str],
    tensor_files: Sequence[str],
) -> None:
    lines: List[str] = []
    lines.append("# BEV Map Prediction Analysis")
    lines.append("")
    lines.append("## Inputs")
    lines.append("")
    lines.append(f"- predictions: `{predictions_path}`")
    if info_path is not None:
        lines.append(f"- infos: `{info_path}`")
    lines.append(f"- samples: `{sample_count}`")
    lines.append(f"- mask shape: `{tuple(mask_shape)}`")
    lines.append(f"- classes: `{', '.join(classes)}`")
    lines.append(f"- thresholds: `{', '.join(f'{x:.2f}' for x in thresholds)}`")
    lines.append("")
    lines.append("## Formula")
    lines.append("")
    lines.append("For each class and threshold:")
    lines.append("")
    lines.append("```text")
    lines.append("pred_binary = masks_bev >= threshold")
    lines.append("label_binary = gt_masks_bev.astype(bool)")
    lines.append("TP = count(pred_binary and label_binary)")
    lines.append("FP = count(pred_binary and not label_binary)")
    lines.append("FN = count(not pred_binary and label_binary)")
    lines.append("IoU = TP / (TP + FP + FN + 1e-7)")
    lines.append("iou@max = max IoU over the configured threshold sweep")
    lines.append("map/mean/iou@max = mean class iou@max over classes with gt_pixels > 0")
    lines.append("```")
    lines.append("")
    lines.append("`masks_bev` is the model probability map returned by the segmentation head at eval time. ")
    lines.append("`gt_masks_bev` is the BEV binary target rasterized by the data pipeline.")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| Class | GT pixels | Best threshold | TP | FP | FN | Union | IoU max |")
    lines.append("| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in class_rows:
        lines.append(
            "| {class_name} | {gt_pixels} | {threshold:.2f} | {tp} | {fp} | {fn} | {union} | {iou:.6f} |".format(
                **row
            )
        )
    lines.append("")
    lines.append(f"- `map/mean/iou@max`: `{metrics['map/mean/iou@max']:.8f}`")
    lines.append(f"- `map/mean/evaluable_classes`: `{int(metrics['map/mean/evaluable_classes'])}`")
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    for artifact in artifacts:
        lines.append(f"- `{artifact}`")
    if image_files:
        lines.append("")
        lines.append("## Visual Checks")
        lines.append("")
        lines.append("Overlay colors: TP green, FP red, FN blue.")
        for image_file in image_files[:30]:
            lines.append(f"- `{image_file}`")
        if len(image_files) > 30:
            lines.append(f"- ... and {len(image_files) - 30} more images")
    lines.append("")
    lines.append("## Selected Tensor Dumps")
    lines.append("")
    lines.append("For selected samples, `selected_tensors/*.npz` contains:")
    lines.append("")
    lines.append("- `masks_bev`: raw prediction probability tensor `[C,H,W]`")
    lines.append("- `gt_masks_bev`: binary ground-truth tensor `[C,H,W]`")
    lines.append("- `pred_binary_best`: prediction thresholded at each class best threshold")
    lines.append("- `match_codes_best`: 0=TN/background, 1=TP, 2=FP, 3=FN")
    if tensor_files:
        lines.append("")
        for tensor_file in tensor_files[:30]:
            lines.append(f"- `{tensor_file}`")
        if len(tensor_files) > 30:
            lines.append(f"- ... and {len(tensor_files) - 30} more tensor dumps")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def analyze(args: argparse.Namespace) -> Dict[str, Any]:
    predictions_path = args.predictions_pkl
    if not predictions_path.exists():
        raise FileNotFoundError(predictions_path)

    classes = tuple(args.classes)
    thresholds = np.asarray(args.thresholds, dtype=np.float32)
    if thresholds.ndim != 1 or len(thresholds) == 0:
        raise ValueError("--thresholds must contain at least one value")

    outputs = load_pickle(predictions_path)
    if not isinstance(outputs, list):
        raise ValueError(f"{predictions_path} must contain a list of result dicts")
    if args.max_samples is not None:
        outputs = outputs[: args.max_samples]
    if not outputs:
        raise RuntimeError("No outputs found")

    infos = read_infos(args.info_pkl)

    num_classes = len(classes)
    num_thresholds = len(thresholds)
    tp = np.zeros((num_classes, num_thresholds), dtype=np.int64)
    fp = np.zeros((num_classes, num_thresholds), dtype=np.int64)
    fn = np.zeros((num_classes, num_thresholds), dtype=np.int64)
    gt_pixels = np.zeros(num_classes, dtype=np.int64)
    pred_pixels = np.zeros((num_classes, num_thresholds), dtype=np.int64)
    prob_sum = np.zeros(num_classes, dtype=np.float64)
    prob_min = np.full(num_classes, np.inf, dtype=np.float64)
    prob_max = np.full(num_classes, -np.inf, dtype=np.float64)
    hist_bins = np.linspace(0.0, 1.0, args.histogram_bins + 1, dtype=np.float64)
    prob_hist = np.zeros((num_classes, args.histogram_bins), dtype=np.int64)

    sample_threshold_rows: List[dict] = []
    sample_scores: List[dict] = []
    mask_shape: Optional[Tuple[int, int, int]] = None

    for sample_index, result in enumerate(outputs):
        if "masks_bev" not in result or "gt_masks_bev" not in result:
            raise KeyError("Each result must contain 'masks_bev' and 'gt_masks_bev'")
        prediction = to_numpy(result["masks_bev"]).astype(np.float32)
        label = to_numpy(result["gt_masks_bev"]).astype(bool)
        validate_shapes(prediction, label, classes)
        if mask_shape is None:
            mask_shape = tuple(int(x) for x in prediction.shape)
        elif tuple(prediction.shape) != mask_shape:
            raise ValueError(
                f"inconsistent mask shape at sample {sample_index}: {prediction.shape} vs {mask_shape}"
            )

        token, timestamp = sample_identity(sample_index, infos)
        sample_iou_accum: List[float] = []

        for class_index, class_name in enumerate(classes):
            pred_flat = prediction[class_index].reshape(-1)
            label_flat = label[class_index].reshape(-1)
            class_gt_pixels = int(label_flat.sum())
            gt_pixels[class_index] += class_gt_pixels
            prob_sum[class_index] += float(pred_flat.sum())
            prob_min[class_index] = min(prob_min[class_index], float(pred_flat.min()))
            prob_max[class_index] = max(prob_max[class_index], float(pred_flat.max()))
            prob_hist[class_index] += np.histogram(pred_flat, bins=hist_bins)[0]

            class_sample_ious: List[float] = []
            for threshold_index, threshold in enumerate(thresholds):
                pred_binary = pred_flat >= threshold
                class_tp = int(np.logical_and(pred_binary, label_flat).sum())
                class_fp = int(np.logical_and(pred_binary, ~label_flat).sum())
                class_fn = int(np.logical_and(~pred_binary, label_flat).sum())
                class_pred_pixels = int(pred_binary.sum())
                union = class_tp + class_fp + class_fn
                iou = float(class_tp / (union + 1e-7))

                tp[class_index, threshold_index] += class_tp
                fp[class_index, threshold_index] += class_fp
                fn[class_index, threshold_index] += class_fn
                pred_pixels[class_index, threshold_index] += class_pred_pixels
                class_sample_ious.append(iou)

                sample_threshold_rows.append(
                    {
                        "sample_index": sample_index,
                        "token": token,
                        "timestamp": "" if timestamp is None else timestamp,
                        "class": class_name,
                        "threshold": f"{float(threshold):.2f}",
                        "gt_pixels": class_gt_pixels,
                        "pred_pixels": class_pred_pixels,
                        "tp": class_tp,
                        "fp": class_fp,
                        "fn": class_fn,
                        "union": union,
                        "iou": format_float(iou),
                        "pred_min": format_float(float(pred_flat.min()), 6),
                        "pred_mean": format_float(float(pred_flat.mean()), 6),
                        "pred_p50": format_float(float(np.percentile(pred_flat, 50)), 6),
                        "pred_p95": format_float(float(np.percentile(pred_flat, 95)), 6),
                        "pred_max": format_float(float(pred_flat.max()), 6),
                    }
                )

            if class_gt_pixels > 0:
                sample_iou_accum.append(max(class_sample_ious))

        mean_best_iou = float(np.mean(sample_iou_accum)) if sample_iou_accum else 0.0
        sample_scores.append(
            {
                "index": sample_index,
                "token": token,
                "timestamp": timestamp,
                "mean_best_iou": mean_best_iou,
            }
        )

    ious = compute_iou(tp, fp, fn)
    valid_classes = gt_pixels > 0
    if not valid_classes.any():
        raise RuntimeError("Every gt_masks_bev label is empty; IoU is undefined")

    class_best_threshold_indices = np.argmax(ious, axis=1)

    metrics: Dict[str, float] = {}
    class_threshold_rows: List[dict] = []
    class_summary_rows: List[dict] = []
    for class_index, class_name in enumerate(classes):
        metrics[f"map/{class_name}/gt_pixels"] = float(gt_pixels[class_index])
        best_index = int(class_best_threshold_indices[class_index])
        if valid_classes[class_index]:
            metrics[f"map/{class_name}/iou@max"] = float(ious[class_index, best_index])
        total_pixels = int(prob_hist[class_index].sum())
        prob_mean = float(prob_sum[class_index] / max(total_pixels, 1))
        prob_p50 = percentile_from_hist(prob_hist[class_index], hist_bins, 50)
        prob_p95 = percentile_from_hist(prob_hist[class_index], hist_bins, 95)
        prob_p99 = percentile_from_hist(prob_hist[class_index], hist_bins, 99)

        for threshold_index, threshold in enumerate(thresholds):
            metric_key = f"map/{class_name}/iou@{float(threshold):.2f}"
            metrics[metric_key] = float(ious[class_index, threshold_index])
            union = int(tp[class_index, threshold_index] + fp[class_index, threshold_index] + fn[class_index, threshold_index])
            class_threshold_rows.append(
                {
                    "class": class_name,
                    "threshold": f"{float(threshold):.2f}",
                    "gt_pixels": int(gt_pixels[class_index]),
                    "pred_pixels": int(pred_pixels[class_index, threshold_index]),
                    "tp": int(tp[class_index, threshold_index]),
                    "fp": int(fp[class_index, threshold_index]),
                    "fn": int(fn[class_index, threshold_index]),
                    "union": union,
                    "iou": format_float(float(ious[class_index, threshold_index])),
                    "is_best_threshold": threshold_index == best_index,
                    "pred_min": format_float(float(prob_min[class_index]), 6),
                    "pred_mean": format_float(prob_mean, 6),
                    "pred_p50_approx": format_float(prob_p50, 6),
                    "pred_p95_approx": format_float(prob_p95, 6),
                    "pred_p99_approx": format_float(prob_p99, 6),
                    "pred_max": format_float(float(prob_max[class_index]), 6),
                }
            )

        best_union = int(tp[class_index, best_index] + fp[class_index, best_index] + fn[class_index, best_index])
        class_summary_rows.append(
            {
                "class_name": class_name,
                "gt_pixels": int(gt_pixels[class_index]),
                "threshold": float(thresholds[best_index]),
                "tp": int(tp[class_index, best_index]),
                "fp": int(fp[class_index, best_index]),
                "fn": int(fn[class_index, best_index]),
                "union": best_union,
                "iou": float(ious[class_index, best_index]),
            }
        )

    metrics["map/mean/iou@max"] = float(ious[valid_classes].max(axis=1).mean())
    metrics["map/mean/evaluable_classes"] = float(valid_classes.sum())

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_json = {
        "inputs": {
            "predictions_pkl": str(predictions_path),
            "info_pkl": None if args.info_pkl is None else str(args.info_pkl),
        },
        "classes": list(classes),
        "thresholds": [float(x) for x in thresholds],
        "samples": len(outputs),
        "mask_shape": list(mask_shape or []),
        "metrics": metrics,
        "class_best": class_summary_rows,
        "notes": {
            "prediction": "masks_bev is expected to be a probability tensor in [0,1].",
            "ground_truth": "gt_masks_bev is converted to bool before evaluation.",
            "formula": "IoU = TP / (TP + FP + FN + 1e-7), thresholded per class.",
        },
    }

    (output_dir / "summary_metrics.json").write_text(
        json.dumps(summary_json, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    write_csv(
        output_dir / "class_thresholds.csv",
        class_threshold_rows,
        [
            "class",
            "threshold",
            "gt_pixels",
            "pred_pixels",
            "tp",
            "fp",
            "fn",
            "union",
            "iou",
            "is_best_threshold",
            "pred_min",
            "pred_mean",
            "pred_p50_approx",
            "pred_p95_approx",
            "pred_p99_approx",
            "pred_max",
        ],
    )

    write_csv(
        output_dir / "sample_thresholds.csv",
        sample_threshold_rows,
        [
            "sample_index",
            "token",
            "timestamp",
            "class",
            "threshold",
            "gt_pixels",
            "pred_pixels",
            "tp",
            "fp",
            "fn",
            "union",
            "iou",
            "pred_min",
            "pred_mean",
            "pred_p50",
            "pred_p95",
            "pred_max",
        ],
    )

    write_csv(
        output_dir / "sample_scores.csv",
        [
            {
                "sample_index": row["index"],
                "token": row["token"],
                "timestamp": "" if row["timestamp"] is None else row["timestamp"],
                "mean_best_iou": format_float(row["mean_best_iou"]),
            }
            for row in sample_scores
        ],
        ["sample_index", "token", "timestamp", "mean_best_iou"],
    )

    requested_indices = parse_indices(args.sample_indices)
    viz_indices = choose_visualization_indices(sample_scores, requested_indices, args.max_viz)
    tensor_files = write_selected_tensor_npz(
        outputs,
        output_dir,
        classes,
        thresholds,
        class_best_threshold_indices,
        viz_indices,
        infos,
    )
    image_files = write_visualizations(
        outputs,
        output_dir,
        classes,
        thresholds,
        class_best_threshold_indices,
        viz_indices,
        infos,
        args.viz_threshold,
        args.viz_scale,
    )

    artifacts = [
        "summary_metrics.json",
        "class_thresholds.csv",
        "sample_thresholds.csv",
        "sample_scores.csv",
        "analysis.md",
        "selected_tensors/*.npz",
    ]
    write_markdown(
        output_dir / "analysis.md",
        predictions_path,
        args.info_pkl,
        classes,
        thresholds,
        len(outputs),
        mask_shape or (),
        metrics,
        class_summary_rows,
        artifacts,
        image_files,
        tensor_files,
    )

    return {
        "output_dir": str(output_dir),
        "metrics": metrics,
        "image_files": image_files,
        "tensor_files": tensor_files,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions-pkl",
        type=Path,
        default=Path("bevfusion-main/runs/vinfast-map-seg/predictions.pkl"),
        help="Pickle produced by tools/test.py --out.",
    )
    parser.add_argument(
        "--info-pkl",
        type=Path,
        help="Optional nuscenes_infos_val.pkl for sample token/timestamp labels.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("bevfusion-main/runs/vinfast-map-seg/metric_analysis"),
        help="Directory for JSON/CSV/Markdown/PNG outputs.",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=list(DEFAULT_CLASSES),
        help="Map class order in masks_bev and gt_masks_bev.",
    )
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=list(DEFAULT_THRESHOLDS),
        help="Threshold sweep. Defaults match NuScenesDataset.evaluate_map.",
    )
    parser.add_argument(
        "--max-viz",
        type=int,
        default=12,
        help="Maximum number of samples selected for visualization.",
    )
    parser.add_argument(
        "--sample-indices",
        help="Comma-separated sample indices to visualize, e.g. 0,10,123.",
    )
    parser.add_argument(
        "--viz-threshold",
        type=float,
        help="Optional fixed visualization threshold. Defaults to each class best threshold.",
    )
    parser.add_argument(
        "--viz-scale",
        type=int,
        default=3,
        help="Nearest-neighbor scale factor for PNG panels.",
    )
    parser.add_argument(
        "--histogram-bins",
        type=int,
        default=100,
        help="Bins for approximate prediction percentiles.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Optional debug limit for the number of outputs to analyze.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = analyze(args)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
