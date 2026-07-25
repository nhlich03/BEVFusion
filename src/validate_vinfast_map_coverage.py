#!/usr/bin/env python3
"""Measure semantic HD-map coverage for VinFast sensor and NAV timestamps."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import pickle
from pathlib import Path

import numpy as np
from pyproj import Transformer
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union


def quaternion_matrix(q):
    w, x, y, z = np.asarray(q, dtype=np.float64) / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/vinfast-nuscenes"))
    parser.add_argument("--vinfast-root", type=Path, default=Path("VinFast - data sample"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    map_path = args.dataset_root / "maps" / "expansion" / "boston-seaport.json"
    map_data = json.loads(map_path.read_text(encoding="utf-8"))
    node_by_token = {item["token"]: (item["x"], item["y"]) for item in map_data["node"]}
    polygon_by_token = {
        item["token"]: Polygon([node_by_token[x] for x in item["exterior_node_tokens"]])
        for item in map_data["polygon"]
    }
    drivable = unary_union(
        [polygon_by_token[item["polygon_token"]] for item in map_data["lane"]]
    )

    infos = []
    for split in ("train", "val"):
        with (args.dataset_root / f"nuscenes_infos_{split}.pkl").open("rb") as handle:
            infos.extend(pickle.load(handle)["infos"])
    sensor_distances = []
    for info in infos:
        ego_rotation = quaternion_matrix(info["ego2global_rotation"])
        lidar_position = (
            np.asarray(info["ego2global_translation"])
            + ego_rotation @ np.asarray(info["lidar2ego_translation"])
        )
        sensor_distances.append(drivable.distance(Point(*lidar_position[:2])))

    origin_data = json.loads(
        (args.dataset_root / "maps" / "expansion" / "vinfast_map_origin.json").read_text(
            encoding="utf-8"
        )
    )
    origin_x, origin_y = origin_data["origin_utm"]
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32648", always_xy=True)
    nav_rows = []
    for path in glob.glob(str(args.vinfast_root / "NAV" / "*.csv")):
        with open(path, "r", encoding="utf-8-sig", newline="") as handle:
            nav_rows.extend(csv.DictReader(handle))
    nav_rows.sort(key=lambda row: row["Timestamp"])

    sensor_start = min(info["timestamp"] for info in infos)
    sensor_end = max(info["timestamp"] for info in infos)
    first_within_50m = None
    first_on_drivable = None
    for row in nav_rows:
        timestamp_us = int(row["Timestamp"].split("-")[0]) * 1_000_000 + int(
            row["Timestamp"].split("-")[1][:6]
        )
        if timestamp_us < sensor_start:
            continue
        x, y = transformer.transform(float(row["Longitude"]), float(row["Latitude"]))
        point = Point(x - origin_x, y - origin_y)
        distance = drivable.distance(point)
        record = {
            "timestamp": row["Timestamp"],
            "seconds_after_sensor_end": (timestamp_us - sensor_end) / 1e6,
            "distance_to_drivable_m": distance,
            "local_xy_m": [x - origin_x, y - origin_y],
            "patch_yaw_deg": 90.0 - float(row["Heading"]),
        }
        if first_within_50m is None and distance <= 50.0:
            first_within_50m = record
        if first_on_drivable is None and drivable.covers(point):
            first_on_drivable = record
        if first_within_50m is not None and first_on_drivable is not None:
            break

    conversion = json.loads(
        (args.dataset_root / "conversion_summary.json").read_text(encoding="utf-8")
    )
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/vinfast-map-validation-mpl")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp/vinfast-map-validation-cache")
    from nuscenes.map_expansion.map_api import NuScenesMap

    probe = None
    if first_on_drivable is not None:
        nusc_map = NuScenesMap(str(args.dataset_root), "boston-seaport")
        px, py = first_on_drivable["local_xy_m"]
        masks = nusc_map.get_map_mask(
            (px, py, 100.0, 100.0),
            first_on_drivable["patch_yaw_deg"],
            ["drivable_area", "ped_crossing", "road_divider", "lane_divider"],
            (200, 200),
        )
        probe = {
            "note": "NAV-only pose; no camera/LiDAR files are available at this timestamp",
            "timestamp": first_on_drivable["timestamp"],
            "positive_pixels": {
                "drivable_area": int(masks[0].sum()),
                "ped_crossing": int(masks[1].sum()),
                "divider": int(np.logical_or(masks[2], masks[3]).sum()),
            },
        }
    report = {
        "evaluation_ready": conversion["evaluation_ready"],
        "sensor_samples": len(infos),
        "sensor_time_range_us": [sensor_start, sensor_end],
        "sensor_distance_to_drivable_m": {
            "min": float(np.min(sensor_distances)),
            "max": float(np.max(sensor_distances)),
            "mean": float(np.mean(sensor_distances)),
        },
        "default_bev_positive_pixels": conversion["labels"]["positive_pixels"],
        "first_nav_pose_within_50m": first_within_50m,
        "first_nav_pose_on_drivable_area": first_on_drivable,
        "map_rasterization_probe": probe,
        "conclusion": (
            "The vector map and long NAV route align, but the available 5-second sensor sample ends "
            "before the route reaches the mapped lane geometry. More camera/LiDAR frames are required."
        ),
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    output = args.output or (args.dataset_root / "map_coverage_report.json")
    output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
