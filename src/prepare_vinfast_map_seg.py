#!/usr/bin/env python3
"""Prepare the VinFast sample as a nuScenes-style BEV map segmentation dataset.

This converter deliberately does not create object-detection labels.  It builds:

* six perspective camera streams and LIDAR_TOP;
* GPS/INS ego poses in the same local UTM frame as the HD map;
* a nuScenes map expansion JSON containing drivable area, crossings and dividers;
* empty instance/sample_annotation tables (schema compatibility only);
* BEVFusion info PKLs with empty box arrays;
* per-sample ``gt_masks_bev`` NPZ files for inspection and independent evaluation.

The BEVFusion loader rasterizes the same map online.  The saved NPZ labels are a
reproducible QA artifact, not an alternate label source.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import os
import pickle
import shutil
import sys
import uuid
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from pyproj import Transformer


CAMERA_MAP = {
    "CAM_FRONT": "CAM_P_F",
    "CAM_FRONT_RIGHT": "CAM_P_FR",
    "CAM_BACK_RIGHT": "CAM_P_RB",
    "CAM_BACK": "CAM_P_B",
    "CAM_BACK_LEFT": "CAM_P_LB",
    "CAM_FRONT_LEFT": "CAM_P_FL",
}
MERGE_LIDAR_CHANNELS = ("LIDAR_TOP", "LIDAR_E_F", "LIDAR_E_L", "LIDAR_E_R", "LIDAR_E_B")
MAP_CLASSES = ("drivable_area", "ped_crossing", "divider")
VERSION = "v1.0-mini"
MAP_NAME = "boston-seaport"  # Must be one of the names accepted by nuScenes-devkit.
TOKEN_NAMESPACE = uuid.UUID("b61068ce-5212-4ac7-84c8-f1306becd2cb")


def token(name: str) -> str:
    return uuid.uuid5(TOKEN_NAMESPACE, name).hex


def parse_timestamp_ns(value: str) -> int:
    stem = Path(str(value)).stem
    sec, frac = stem.split("-", 1)
    return int(sec) * 1_000_000_000 + int(frac[:9].ljust(9, "0"))


def qmul(a: Sequence[float], b: Sequence[float]) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=np.float64,
    )


def axis_quaternion(axis: int, degrees: float) -> np.ndarray:
    half = math.radians(degrees) / 2.0
    out = np.zeros(4, dtype=np.float64)
    out[0] = math.cos(half)
    out[axis + 1] = math.sin(half)
    return out


def sensor_quaternion(euler_xyz_deg: Sequence[float]) -> np.ndarray:
    """VinFast extrinsics use the intrinsic X-Y-Z order (camera -> ego)."""
    qx = axis_quaternion(0, float(euler_xyz_deg[0]))
    qy = axis_quaternion(1, float(euler_xyz_deg[1]))
    qz = axis_quaternion(2, float(euler_xyz_deg[2]))
    q = qmul(qmul(qx, qy), qz)
    return q / np.linalg.norm(q)


def ego_quaternion(roll: float, pitch: float, heading: float) -> np.ndarray:
    """Convert clockwise-from-North heading to ENU yaw (CCW from East)."""
    yaw = 90.0 - heading
    q = qmul(
        qmul(axis_quaternion(2, yaw), axis_quaternion(1, pitch)),
        axis_quaternion(0, roll),
    )
    return q / np.linalg.norm(q)


def quaternion_matrix(q: Sequence[float]) -> np.ndarray:
    w, x, y, z = np.asarray(q, dtype=np.float64) / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def json_dump(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def clean_ring(ids: Iterable[str]) -> List[str]:
    out: List[str] = []
    for item in ids:
        if not out or item != out[-1]:
            out.append(item)
    if out and out[-1] != out[0]:
        out.append(out[0])
    return out


def build_map(osm_path: Path) -> Tuple[dict, Tuple[float, float], dict]:
    root = ET.parse(osm_path).getroot()
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32648", always_xy=True)

    raw_nodes: Dict[str, Tuple[float, float]] = {}
    latitudes: List[float] = []
    longitudes: List[float] = []
    for node in root.findall("node"):
        lat = float(node.attrib["lat"])
        lon = float(node.attrib["lon"])
        raw_nodes[node.attrib["id"]] = transformer.transform(lon, lat)
        latitudes.append(lat)
        longitudes.append(lon)

    origin_x = min(x for x, _ in raw_nodes.values())
    origin_y = min(y for _, y in raw_nodes.values())
    local_nodes = {key: (x - origin_x, y - origin_y) for key, (x, y) in raw_nodes.items()}
    ways = {
        way.attrib["id"]: [nd.attrib["ref"] for nd in way.findall("nd")]
        for way in root.findall("way")
    }

    node_token = {node_id: token(f"map-node:{node_id}") for node_id in local_nodes}
    way_token = {way_id: token(f"map-line:{way_id}") for way_id in ways}
    nodes = [
        {"token": node_token[node_id], "x": round(x, 3), "y": round(y, 3)}
        for node_id, (x, y) in local_nodes.items()
    ]
    lines = [
        {"token": way_token[way_id], "node_tokens": [node_token[x] for x in refs]}
        for way_id, refs in ways.items()
    ]

    polygons: List[dict] = []
    lanes: List[dict] = []
    crossings: List[dict] = []
    road_segments: List[dict] = []
    road_polygon_tokens: List[str] = []
    road_way_usage: Counter = Counter()
    relation_stats: Counter = Counter()

    def distance(a: str, b: str) -> float:
        ax, ay = local_nodes[a]
        bx, by = local_nodes[b]
        return math.hypot(ax - bx, ay - by)

    for relation in root.findall("relation"):
        tags = {tag.attrib["k"]: tag.attrib["v"] for tag in relation.findall("tag")}
        subtype = tags.get("subtype")
        if tags.get("type") != "lanelet" or subtype not in {"road", "crosswalk"}:
            continue
        members = {
            member.attrib.get("role"): member.attrib.get("ref")
            for member in relation.findall("member")
            if member.attrib.get("type") == "way"
        }
        if not members.get("left") or not members.get("right"):
            continue
        left_id = members["left"]
        right_id = members["right"]
        left = list(dict.fromkeys(ways[left_id]))
        right = list(dict.fromkeys(ways[right_id]))

        # Lanelet boundaries normally have the same direction.  Some source
        # relations do not, so choose the orientation with the shorter end caps.
        same_direction = distance(left[-1], right[-1]) + distance(right[0], left[0])
        opposite_direction = distance(left[-1], right[0]) + distance(right[-1], left[0])
        if opposite_direction < same_direction:
            right.reverse()
        ring = clean_ring(left + list(reversed(right)))

        relation_id = relation.attrib["id"]
        polygon_token = token(f"map-polygon:{relation_id}")
        polygons.append(
            {
                "token": polygon_token,
                "exterior_node_tokens": [node_token[x] for x in ring],
                "holes": [],
            }
        )
        relation_stats[subtype] += 1

        if subtype == "road":
            # lane_token = token(f"map-lane:{relation_id}")
            # lanes.append(
            #     {
            #         "token": lane_token,
            #         "polygon_token": polygon_token,
            #         "lane_type": "CAR",
            #         "from_edge_line_token": "",
            #         "to_edge_line_token": "",
            #         "left_lane_divider_segments": [],
            #         "right_lane_divider_segments": [],
            #         "speed_limit": float(tags.get("speed_limit", 0.0)),
            #         "is_one_way": tags.get("one_way") == "yes",
            #     }
            # )
            road_segments.append(
                {
                    "token": token(f"map-road-segment:{relation_id}"),
                    "polygon_token": polygon_token,
                    "is_intersection": tags.get("turn_direction", "straight") != "straight",
                }
            )
            road_polygon_tokens.append(polygon_token)
            road_way_usage[left_id] += 1
            road_way_usage[right_id] += 1
        else:
            crossings.append(
                {
                    "token": token(f"map-crossing:{relation_id}"),
                    "polygon_token": polygon_token,
                    "road_segment_token": "",
                }
            )

    road_dividers: List[dict] = []
    lane_dividers: List[dict] = []
    for way_id, usage in road_way_usage.items():
        record = {
            "token": token(f"map-divider:{way_id}"),
            "line_token": way_token[way_id],
            "divider_type": "SOLID_LINE" if usage == 1 else "DASHED_LINE",
        }
        (lane_dividers if usage > 1 else road_dividers).append(record)

    max_x = max(x for x, _ in local_nodes.values())
    max_y = max(y for _, y in local_nodes.values())
    map_json = {
        "version": "1.3",
        "canvas_edge": [float(math.ceil(max_x + 1)), float(math.ceil(max_y + 1))],
        "node": nodes,
        "line": lines,
        "polygon": polygons,
        "drivable_area": [
            {"token": token("map-drivable-area:all-roads"), "polygon_tokens": road_polygon_tokens}
        ],
        "road_segment": road_segments,
        "road_block": [],
        "lane": lanes,
        "ped_crossing": crossings,
        "walkway": [],
        "stop_line": [],
        "carpark_area": [],
        "road_divider": road_dividers,
        "lane_divider": lane_dividers,
        "traffic_light": [],
        "lane_connector": [],
        "arcline_path_3": {},
        "connectivity": {},
    }
    metadata = {
        "projection": "EPSG:32648",
        "origin_utm": [origin_x, origin_y],
        "origin_rule": "minimum UTM easting/northing over all OSM nodes",
        "source_bbox_wgs84": [
            min(latitudes),
            min(longitudes),
            max(latitudes),
            max(longitudes),
        ],
        "local_extent_m": [max_x, max_y],
        "layers": {
            "road_lanelets": relation_stats["road"],
            "ped_crossings": relation_stats["crosswalk"],
            "road_dividers": len(road_dividers),
            "lane_dividers": len(lane_dividers),
        },
    }
    return map_json, (origin_x, origin_y), metadata


class Navigation:
    def __init__(self, nav_dir: Path, map_origin: Tuple[float, float]):
        rows: List[dict] = []
        for path in sorted(nav_dir.glob("*.csv")):
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                rows.extend(csv.DictReader(handle))
        rows.sort(key=lambda row: parse_timestamp_ns(row["Timestamp"]))
        if not rows:
            raise RuntimeError(f"No NAV rows found in {nav_dir}")

        self.timestamps = np.asarray(
            [parse_timestamp_ns(row["Timestamp"]) for row in rows], dtype=np.int64
        )
        self.lat = np.asarray([float(row["Latitude"]) for row in rows])
        self.lon = np.asarray([float(row["Longitude"]) for row in rows])
        self.alt = np.asarray([float(row["Altitude"]) for row in rows])
        self.roll = np.asarray([float(row["Roll"]) for row in rows])
        self.pitch = np.asarray([float(row["Pitch"]) for row in rows])
        heading = np.asarray([float(row["Heading"]) for row in rows])
        self.heading_unwrapped = np.unwrap(np.radians(heading))
        self.transformer = Transformer.from_crs("EPSG:4326", "EPSG:32648", always_xy=True)
        self.origin_x, self.origin_y = map_origin
        self.altitude_origin: float | None = None

    def _interp(self, values: np.ndarray, timestamp_ns: int) -> float:
        return float(np.interp(timestamp_ns, self.timestamps, values))

    def pose(self, timestamp_ns: int) -> Tuple[np.ndarray, np.ndarray, dict]:
        lat = self._interp(self.lat, timestamp_ns)
        lon = self._interp(self.lon, timestamp_ns)
        alt = self._interp(self.alt, timestamp_ns)
        if self.altitude_origin is None:
            self.altitude_origin = alt
        roll = self._interp(self.roll, timestamp_ns)
        pitch = self._interp(self.pitch, timestamp_ns)
        heading = math.degrees(self._interp(self.heading_unwrapped, timestamp_ns)) % 360.0
        x, y = self.transformer.transform(lon, lat)
        translation = np.array(
            [x - self.origin_x, y - self.origin_y, alt - self.altitude_origin], dtype=np.float64
        )
        rotation = ego_quaternion(roll, pitch, heading)
        return translation, rotation, {
            "latitude": lat,
            "longitude": lon,
            "altitude": alt,
            "roll": roll,
            "pitch": pitch,
            "heading": heading,
        }


def nearest_file(files: Sequence[Tuple[int, Path]], timestamp_ns: int) -> Tuple[int, Path]:
    times = [item[0] for item in files]
    index = bisect.bisect_left(times, timestamp_ns)
    candidates = []
    if index < len(files):
        candidates.append(files[index])
    if index:
        candidates.append(files[index - 1])
    return min(candidates, key=lambda item: abs(item[0] - timestamp_ns))


def output_relpath(path: Path, output_root: Path) -> str:
    return path.relative_to(output_root).as_posix()


def resolve_source_dir(source_root: Path, *relative_options: Sequence[str]) -> Path:
    for relative in relative_options:
        path = source_root.joinpath(*relative)
        if path.is_dir():
            return path
    tried = ", ".join("/".join(parts) for parts in relative_options)
    raise RuntimeError(f"Cannot find any source directory under {source_root}: {tried}")


def calibration_from_extrinsics(extrinsics: dict, source_name: str) -> dict:
    values = extrinsics[source_name]
    return {
        "translation": np.asarray(values[:3], dtype=np.float64) / 1000.0,
        "rotation": sensor_quaternion(values[3:6]),
        "source_name": source_name,
    }


def sensor_to_reference(
    sensor_translation: np.ndarray,
    sensor_rotation: np.ndarray,
    sensor_ego_translation: np.ndarray,
    sensor_ego_rotation: np.ndarray,
    reference_translation: np.ndarray,
    reference_rotation: np.ndarray,
    reference_ego_translation: np.ndarray,
    reference_ego_rotation: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    sensor_global_rotation = quaternion_matrix(sensor_ego_rotation) @ quaternion_matrix(sensor_rotation)
    reference_global_rotation = quaternion_matrix(reference_ego_rotation) @ quaternion_matrix(reference_rotation)
    sensor_global_translation = (
        quaternion_matrix(sensor_ego_rotation) @ sensor_translation + sensor_ego_translation
    )
    reference_global_translation = (
        quaternion_matrix(reference_ego_rotation) @ reference_translation + reference_ego_translation
    )
    rotation = reference_global_rotation.T @ sensor_global_rotation
    translation = reference_global_rotation.T @ (sensor_global_translation - reference_global_translation)
    return rotation, translation


def materialize_image(source: Path, destination: Path, intrinsic: dict, undistort: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if undistort and destination.is_symlink():
        # Replace a previous space-saving link with a calibrated image when
        # the converter is rerun with --undistort-images.
        destination.unlink()
    if destination.exists() or destination.is_symlink():
        return
    if not undistort:
        destination.symlink_to(os.path.relpath(source, destination.parent))
        return
    import cv2

    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Cannot read image {source}")
    camera_matrix = np.asarray(intrinsic["camera_matrix"], dtype=np.float64)
    distortion = np.asarray(intrinsic["distortion_coefficients"], dtype=np.float64)
    output = cv2.undistort(image, camera_matrix, distortion, None, camera_matrix)
    if not cv2.imwrite(str(destination), output, [cv2.IMWRITE_JPEG_QUALITY, 95]):
        raise RuntimeError(f"Cannot write image {destination}")


def load_lidar_points(source: Path) -> np.ndarray:
    deps = Path(__file__).resolve().parent / ".deps"
    if deps.exists() and str(deps) not in sys.path:
        # numpy is already imported, so the locally installed laspy/lazrs can
        # be used without shadowing the environment's compatible numpy build.
        sys.path.insert(0, str(deps))
    try:
        import laspy
    except ImportError as exc:
        raise RuntimeError("Install LAZ support with: pip install laspy lazrs") from exc
    cloud = laspy.read(source)
    points = np.column_stack(
        [
            np.asarray(cloud.x, dtype=np.float32),
            np.asarray(cloud.y, dtype=np.float32),
            np.asarray(cloud.z, dtype=np.float32),
            np.asarray(cloud.intensity, dtype=np.float32),
            np.zeros(len(cloud.x), dtype=np.float32),
        ]
    ).astype(np.float32, copy=False)
    return points


def convert_lidar(source: Path, destination: Path, overwrite: bool = False) -> int:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        return destination.stat().st_size // (5 * 4)
    points = load_lidar_points(source)
    points.tofile(destination)
    return len(points)


def transform_points(points: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    transformed = points.copy()
    transformed[:, :3] = transformed[:, :3] @ rotation.T + translation
    return transformed


def convert_merged_lidars(
    keyframe_source: Path,
    lidar_sources_by_channel: Dict[str, Sequence[Tuple[int, Path]]],
    calibrations_by_channel: Dict[str, dict],
    reference_channel: str,
    timestamp_ns: int,
    destination: Path,
    max_delta_ms: float,
    overwrite: bool = False,
) -> int:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        return destination.stat().st_size // (5 * 4)

    reference = calibrations_by_channel[reference_channel]
    identity_translation = np.zeros(3, dtype=np.float64)
    identity_rotation = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    point_sets: List[np.ndarray] = []

    for channel, sources in lidar_sources_by_channel.items():
        if channel == reference_channel:
            source = keyframe_source
        else:
            source_ts, source = nearest_file(sources, timestamp_ns)
            delta_ms = abs(source_ts - timestamp_ns) / 1e6
            if delta_ms > max_delta_ms:
                raise RuntimeError(
                    f"{channel} nearest sweep is {delta_ms:.3f} ms from "
                    f"{reference_channel} timestamp {timestamp_ns}; increase "
                    "--max-lidar-merge-delta-ms only if this is expected."
                )

        points = load_lidar_points(source)
        if channel != reference_channel:
            calibration = calibrations_by_channel[channel]
            rotation, translation = sensor_to_reference(
                calibration["translation"],
                calibration["rotation"],
                identity_translation,
                identity_rotation,
                reference["translation"],
                reference["rotation"],
                identity_translation,
                identity_rotation,
            )
            points = transform_points(points, rotation, translation)
        point_sets.append(points)

    merged = np.concatenate(point_sets, axis=0).astype(np.float32, copy=False)
    merged.tofile(destination)
    return len(merged)


def empty_detection_info() -> dict:
    return {
        "gt_boxes": np.empty((0, 7), dtype=np.float32),
        "gt_names": np.empty((0,), dtype="<U1"),
        "gt_velocity": np.empty((0, 2), dtype=np.float32),
        "num_lidar_pts": np.empty((0,), dtype=np.int64),
        "num_radar_pts": np.empty((0,), dtype=np.int64),
        "valid_flag": np.empty((0,), dtype=bool),
    }


def prepare(args: argparse.Namespace) -> dict:
    source_root = args.vinfast_root.resolve()
    output_root = args.output_root.resolve()
    version_dir = output_root / VERSION
    map_dir = output_root / "maps" / "expansion"
    output_root.mkdir(parents=True, exist_ok=True)
    version_dir.mkdir(parents=True, exist_ok=True)
    map_dir.mkdir(parents=True, exist_ok=True)

    osm_path = source_root / "HDMAP-Demo-Region" / "lanelet2_map.osm"
    map_json, map_origin, map_metadata = build_map(osm_path)
    json_dump(map_dir / f"{MAP_NAME}.json", map_json)
    json_dump(map_dir / "vinfast_map_origin.json", map_metadata)

    with (source_root / "Camera_Intrinsics.json").open("r", encoding="utf-8") as handle:
        intrinsics = json.load(handle)
    with (source_root / "Extrinsics.json").open("r", encoding="utf-8") as handle:
        extrinsics = json.load(handle)

    navigation = Navigation(source_root / "NAV", map_origin)
    lidar_dir = resolve_source_dir(
        source_root,
        ("LIDAR", "LIDAR_TOP"),
        ("lidar_sample", "LIDAR_TOP"),
    )
    lidar_dirs: Dict[str, Path] = {"LIDAR_TOP": lidar_dir}
    for channel in MERGE_LIDAR_CHANNELS:
        if channel == "LIDAR_TOP" or channel not in extrinsics:
            continue
        try:
            lidar_dirs[channel] = resolve_source_dir(
                source_root,
                ("LIDAR", channel),
                ("lidar_sample", channel),
            )
        except RuntimeError:
            # Full VinFast exports may not contain every calibrated LiDAR.  Fuse
            # all channels that are physically present, with LIDAR_TOP as the
            # nuScenes-compatible reference frame.
            pass
    camera_dirs = {
        channel: resolve_source_dir(
            source_root,
            ("CAMERA", source_name),
            ("camera_sample", source_name),
        )
        for channel, source_name in CAMERA_MAP.items()
    }

    lidar_sources = sorted(
        (parse_timestamp_ns(path.name), path)
        for path in lidar_dir.glob("*.laz")
    )
    if not lidar_sources:
        raise RuntimeError(f"No LIDAR_TOP .laz files found in {lidar_dir}")
    lidar_sources_by_channel = {"LIDAR_TOP": lidar_sources}
    for channel, channel_dir in lidar_dirs.items():
        if channel == "LIDAR_TOP":
            continue
        sources = sorted(
            (parse_timestamp_ns(path.name), path)
            for path in channel_dir.glob("*.laz")
        )
        if not sources:
            raise RuntimeError(f"No {channel} .laz files found in {channel_dir}")
        lidar_sources_by_channel[channel] = sources

    camera_sources = {
        channel: sorted(
            (parse_timestamp_ns(path.name), path)
            for path in camera_dirs[channel].glob("*.jpg")
        )
        for channel, source_name in CAMERA_MAP.items()
    }
    for channel, files in camera_sources.items():
        if not files:
            raise RuntimeError(f"No images found for {channel} in {camera_dirs[channel]}")

    sensor_records: List[dict] = []
    calibrated_records: List[dict] = []
    calibrations: Dict[str, dict] = {}
    lidar_calibrations = {
        channel: calibration_from_extrinsics(extrinsics, channel)
        for channel in lidar_sources_by_channel
    }
    for channel, source_name in [("LIDAR_TOP", "LIDAR_TOP"), *CAMERA_MAP.items()]:
        modality = "lidar" if channel == "LIDAR_TOP" else "camera"
        sensor_tok = token(f"sensor:{channel}")
        calibration_tok = token(f"calibration:{channel}")
        calibration_values = calibration_from_extrinsics(extrinsics, source_name)
        translation = calibration_values["translation"]
        rotation = calibration_values["rotation"]
        camera_intrinsic = intrinsics[source_name]["camera_matrix"] if modality == "camera" else []
        sensor_records.append(
            {"token": sensor_tok, "channel": channel, "modality": modality}
        )
        calibrated_records.append(
            {
                "token": calibration_tok,
                "sensor_token": sensor_tok,
                "translation": translation.tolist(),
                "rotation": rotation.tolist(),
                "camera_intrinsic": camera_intrinsic,
            }
        )
        calibrations[channel] = {
            "token": calibration_tok,
            "translation": translation,
            "rotation": rotation,
            "source_name": source_name,
        }

    split_index = max(1, min(len(lidar_sources) - 1, round(len(lidar_sources) * args.train_ratio)))
    split_specs = [
        ("train", "scene-0061", lidar_sources[:split_index]),
        ("val", "scene-0103", lidar_sources[split_index:]),
    ]
    log_token = token("log:vinfast-sample")
    scene_records: List[dict] = []
    sample_records: List[dict] = []
    sample_data_records: List[dict] = []
    ego_pose_records: List[dict] = []
    ego_pose_cache: Dict[int, dict] = {}
    infos_by_split: Dict[str, List[dict]] = {"train": [], "val": []}
    frames: List[dict] = []
    sync_errors_ms: List[float] = []
    sync_errors_by_channel_ms: Dict[str, List[float]] = {
        channel: [] for channel in CAMERA_MAP
    }
    point_counts: List[int] = []

    def get_pose(timestamp_ns: int) -> dict:
        if timestamp_ns not in ego_pose_cache:
            translation, rotation, raw = navigation.pose(timestamp_ns)
            record = {
                "token": token(f"ego-pose:{timestamp_ns}"),
                "timestamp": timestamp_ns // 1000,
                "translation": translation.tolist(),
                "rotation": rotation.tolist(),
            }
            ego_pose_cache[timestamp_ns] = {**record, "raw": raw}
            ego_pose_records.append(record)
        return ego_pose_cache[timestamp_ns]

    for split_name, scene_name, split_lidars in split_specs:
        scene_token = token(f"scene:{scene_name}")
        scene_samples: List[dict] = []
        previous_sample_data: Dict[str, dict] = {}
        previous_info_frames: List[dict] = []

        for frame_index, (lidar_ts_ns, lidar_source) in enumerate(split_lidars):
            if frame_index % 5 == 0 or frame_index == len(split_lidars) - 1:
                print(f"[{split_name.upper()}] Converting frame {frame_index + 1}/{len(split_lidars)} (Timestamp: {lidar_ts_ns // 1000000} ms)...", flush=True)
            sample_token = token(f"sample:{lidar_ts_ns}")
            lidar_pose = get_pose(lidar_ts_ns)
            lidar_calibration = calibrations["LIDAR_TOP"]
            lidar_destination = output_root / "samples" / "LIDAR_TOP" / f"{lidar_source.stem}.bin"
            lidar_relpath = output_relpath(lidar_destination, output_root)
            if len(lidar_sources_by_channel) > 1:
                point_counts.append(
                    convert_merged_lidars(
                        lidar_source,
                        lidar_sources_by_channel,
                        lidar_calibrations,
                        "LIDAR_TOP",
                        lidar_ts_ns,
                        lidar_destination,
                        args.max_lidar_merge_delta_ms,
                        overwrite=args.overwrite_lidar,
                    )
                )
            else:
                point_counts.append(
                    convert_lidar(
                        lidar_source,
                        lidar_destination,
                        overwrite=args.overwrite_lidar,
                    )
                )

            sample = {
                "token": sample_token,
                "timestamp": lidar_ts_ns // 1000,
                "prev": scene_samples[-1]["token"] if scene_samples else "",
                "next": "",
                "scene_token": scene_token,
            }
            if scene_samples:
                scene_samples[-1]["next"] = sample_token
            scene_samples.append(sample)
            sample_records.append(sample)

            lidar_sd_token = token(f"sample-data:LIDAR_TOP:{lidar_ts_ns}")
            lidar_record = {
                "token": lidar_sd_token,
                "sample_token": sample_token,
                "ego_pose_token": lidar_pose["token"],
                "calibrated_sensor_token": lidar_calibration["token"],
                "timestamp": lidar_ts_ns // 1000,
                "fileformat": "bin",
                "is_key_frame": True,
                "height": 0,
                "width": 0,
                "filename": lidar_relpath,
                "prev": previous_sample_data.get("LIDAR_TOP", {}).get("token", ""),
                "next": "",
            }
            if "LIDAR_TOP" in previous_sample_data:
                previous_sample_data["LIDAR_TOP"]["next"] = lidar_sd_token
            previous_sample_data["LIDAR_TOP"] = lidar_record
            sample_data_records.append(lidar_record)

            lidar_t = lidar_calibration["translation"]
            lidar_q = lidar_calibration["rotation"]
            lidar_ego_t = np.asarray(lidar_pose["translation"])
            lidar_ego_q = np.asarray(lidar_pose["rotation"])
            camera_infos: Dict[str, dict] = {}

            for channel, source_name in CAMERA_MAP.items():
                camera_ts_ns, camera_source = nearest_file(camera_sources[channel], lidar_ts_ns)
                sync_error_ms = abs(camera_ts_ns - lidar_ts_ns) / 1e6
                if sync_error_ms > args.max_camera_lidar_delta_ms:
                    raise RuntimeError(
                        f"{channel} nearest image is {sync_error_ms:.3f} ms from "
                        f"LIDAR_TOP timestamp {lidar_ts_ns}; increase "
                        "--max-camera-lidar-delta-ms only if this is expected."
                    )
                sync_errors_ms.append(sync_error_ms)
                sync_errors_by_channel_ms[channel].append(sync_error_ms)
                camera_pose = get_pose(camera_ts_ns)
                calibration = calibrations[channel]
                destination = output_root / "samples" / channel / camera_source.name
                camera_relpath = output_relpath(destination, output_root)
                materialize_image(
                    camera_source,
                    destination,
                    intrinsics[source_name],
                    args.undistort_images,
                )
                camera_sd_token = token(f"sample-data:{channel}:{camera_ts_ns}:{sample_token}")
                camera_record = {
                    "token": camera_sd_token,
                    "sample_token": sample_token,
                    "ego_pose_token": camera_pose["token"],
                    "calibrated_sensor_token": calibration["token"],
                    "timestamp": camera_ts_ns // 1000,
                    "fileformat": "jpg",
                    "is_key_frame": True,
                    "height": int(intrinsics[source_name]["image_height"]),
                    "width": int(intrinsics[source_name]["image_width"]),
                    "filename": camera_relpath,
                    "prev": previous_sample_data.get(channel, {}).get("token", ""),
                    "next": "",
                }
                if channel in previous_sample_data:
                    previous_sample_data[channel]["next"] = camera_sd_token
                previous_sample_data[channel] = camera_record
                sample_data_records.append(camera_record)

                rotation, translation = sensor_to_reference(
                    calibration["translation"],
                    calibration["rotation"],
                    np.asarray(camera_pose["translation"]),
                    np.asarray(camera_pose["rotation"]),
                    lidar_t,
                    lidar_q,
                    lidar_ego_t,
                    lidar_ego_q,
                )
                camera_infos[channel] = {
                    "data_path": camera_relpath,
                    "type": channel,
                    "sample_data_token": camera_sd_token,
                    "sensor2ego_translation": calibration["translation"].tolist(),
                    "sensor2ego_rotation": calibration["rotation"].tolist(),
                    "ego2global_translation": camera_pose["translation"],
                    "ego2global_rotation": camera_pose["rotation"],
                    "timestamp": camera_ts_ns // 1000,
                    "sensor2lidar_rotation": rotation,
                    "sensor2lidar_translation": translation,
                    "cam_intrinsic": np.asarray(
                        intrinsics[source_name]["camera_matrix"], dtype=np.float32
                    ),
                }

            sweeps: List[dict] = []
            for previous in reversed(previous_info_frames[-args.max_sweeps :]):
                rotation, translation = sensor_to_reference(
                    lidar_t,
                    lidar_q,
                    previous["ego_translation"],
                    previous["ego_rotation"],
                    lidar_t,
                    lidar_q,
                    lidar_ego_t,
                    lidar_ego_q,
                )
                sweeps.append(
                    {
                        "data_path": previous["lidar_path"],
                        "type": "lidar",
                        "sample_data_token": previous["lidar_sd_token"],
                        "sensor2ego_translation": lidar_t.tolist(),
                        "sensor2ego_rotation": lidar_q.tolist(),
                        "ego2global_translation": previous["ego_translation"].tolist(),
                        "ego2global_rotation": previous["ego_rotation"].tolist(),
                        "timestamp": previous["timestamp_us"],
                        "sensor2lidar_rotation": rotation,
                        "sensor2lidar_translation": translation,
                    }
                )

            info = {
                "lidar_path": lidar_relpath,
                "token": sample_token,
                "sweeps": sweeps,
                "cams": camera_infos,
                "lidar2ego_translation": lidar_t.tolist(),
                "lidar2ego_rotation": lidar_q.tolist(),
                "ego2global_translation": lidar_pose["translation"],
                "ego2global_rotation": lidar_pose["rotation"],
                "timestamp": lidar_ts_ns // 1000,
                "prev_token": scene_samples[-2]["token"] if len(scene_samples) > 1 else "",
                "prev": frame_index - 1,
                "location": MAP_NAME,
                **empty_detection_info(),
            }
            infos_by_split[split_name].append(info)
            frame = {
                "sample_token": sample_token,
                "split": split_name,
                "timestamp_us": lidar_ts_ns // 1000,
                "ego_translation": lidar_ego_t,
                "ego_rotation": lidar_ego_q,
                "lidar_translation": lidar_t,
                "lidar_rotation": lidar_q,
                "lidar_path": lidar_relpath,
                "lidar_sd_token": lidar_sd_token,
            }
            frames.append(frame)
            previous_info_frames.append(frame)

        scene_records.append(
            {
                "token": scene_token,
                "log_token": log_token,
                "nbr_samples": len(scene_samples),
                "first_sample_token": scene_samples[0]["token"],
                "last_sample_token": scene_samples[-1]["token"],
                "name": scene_name,
                "description": f"VinFast map segmentation {split_name} split",
            }
        )

    first_timestamp = lidar_sources[0][0] / 1e9
    table_data = {
        "category": [],
        "attribute": [],
        "visibility": [],
        "sensor": sensor_records,
        "calibrated_sensor": calibrated_records,
        "ego_pose": ego_pose_records,
        "log": [
            {
                "token": log_token,
                "logfile": "vinfast-data-sample",
                "vehicle": "vinfast-research-vehicle",
                "date_captured": datetime.fromtimestamp(first_timestamp, tz=timezone.utc).date().isoformat(),
                "location": MAP_NAME,
            }
        ],
        "scene": scene_records,
        "sample": sample_records,
        "sample_data": sample_data_records,
        "sample_annotation": [],
        "instance": [],
        "map": [
            {
                "category": "semantic_prior",
                "token": token("map-table:vinfast"),
                "filename": "maps/vinfast-map.png",
                "log_tokens": [log_token],
            }
        ],
    }
    for name, records in table_data.items():
        json_dump(version_dir / f"{name}.json", records)

    # NuScenes does not need the bitmap for vector-map segmentation, but the
    # map table points to a real file to keep the dataset schema self-contained.
    bitmap_path = output_root / "maps" / "vinfast-map.png"
    if not bitmap_path.exists():
        from PIL import Image

        Image.new("L", (1, 1), color=0).save(bitmap_path)

    metadata = {
        "version": VERSION,
        "map_classes": list(MAP_CLASSES),
        "task": "map_segmentation_only",
        "object_annotations": "intentionally empty",
    }
    for split_name in ("train", "val"):
        with (output_root / f"nuscenes_infos_{split_name}.pkl").open("wb") as handle:
            pickle.dump(
                {"infos": infos_by_split[split_name], "metadata": metadata},
                handle,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    label_summary = create_bev_labels(output_root, frames, args.label_resolution)
    total_positive_label_pixels = sum(label_summary["positive_pixels"].values())
    evaluation_ready = total_positive_label_pixels > 0
    warnings = []
    if not evaluation_ready:
        warnings.append(
            "All gt_masks_bev are empty in the default 100 m x 100 m BEV scope. "
            "The available LIDAR_TOP timestamps do not overlap the mapped lane coverage; "
            "do not train or report mIoU on this split. Use a raw LIDAR segment whose "
            "ego poses intersect the HD map, or fix the map/pose alignment before evaluation."
        )
    summary = {
        "task": "map_segmentation_only",
        "source_root": str(source_root),
        "output_root": str(output_root),
        "version": VERSION,
        "map_name": MAP_NAME,
        "map_classes": list(MAP_CLASSES),
        "samples": len(frames),
        "train_samples": len(infos_by_split["train"]),
        "val_samples": len(infos_by_split["val"]),
        "source_samples": {
            "lidar_top": len(lidar_sources),
            "lidar": {
                channel: len(files)
                for channel, files in lidar_sources_by_channel.items()
            },
            "camera": {channel: len(files) for channel, files in camera_sources.items()},
        },
        "source_dirs": {
            "lidar_top": str(lidar_dir.relative_to(source_root)),
            "lidar": {
                channel: str(path.relative_to(source_root))
                for channel, path in lidar_dirs.items()
            },
            "camera": {
                channel: str(camera_dirs[channel].relative_to(source_root))
                for channel in CAMERA_MAP
            },
        },
        "lidar_merge": {
            "enabled": len(lidar_sources_by_channel) > 1,
            "reference_channel": "LIDAR_TOP",
            "channels": list(lidar_sources_by_channel),
            "max_allowed_delta_ms": args.max_lidar_merge_delta_ms,
        },
        "alignment": {
            "strategy": "Use each LIDAR_TOP timestamp as a keyframe and attach the nearest image from every camera channel.",
            "selected_samples_per_channel": {
                channel: len(files) for channel, files in sync_errors_by_channel_ms.items()
            },
            "max_allowed_camera_lidar_delta_ms": args.max_camera_lidar_delta_ms,
        },
        "cameras": CAMERA_MAP,
        "camera_sync_error_ms": {
            "max": max(sync_errors_ms),
            "mean": float(np.mean(sync_errors_ms)),
            "by_channel": {
                channel: {
                    "max": max(values),
                    "mean": float(np.mean(values)),
                }
                for channel, values in sync_errors_by_channel_ms.items()
            },
        },
        "lidar_points": {
            "min": min(point_counts),
            "max": max(point_counts),
            "mean": float(np.mean(point_counts)),
        },
        "map": map_metadata,
        "labels": label_summary,
        "evaluation_ready": evaluation_ready,
        "warnings": warnings,
    }
    json_dump(output_root / "conversion_summary.json", summary)
    for warning in warnings:
        print(f"WARNING: {warning}", file=sys.stderr)
    if warnings and not args.allow_empty_labels:
        raise RuntimeError(
            "Generated map labels are empty, so this split cannot be evaluated. "
            "Re-run with --allow-empty-labels only if you are intentionally debugging "
            "the converter output and will not train/evaluate mIoU from it."
        )
    return summary


def create_bev_labels(output_root: Path, frames: Sequence[dict], resolution: float) -> dict:
    os.environ.setdefault("MPLCONFIGDIR", str(output_root / ".mpl-cache"))
    os.environ.setdefault("XDG_CACHE_HOME", str(output_root / ".cache"))
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    from nuscenes.map_expansion.map_api import NuScenesMap

    nusc_map = NuScenesMap(dataroot=str(output_root), map_name=MAP_NAME)
    labels_dir = output_root / "gt_masks_bev"
    labels_dir.mkdir(parents=True, exist_ok=True)
    patch_size = (100.0, 100.0)
    canvas_size = (round(patch_size[0] / resolution), round(patch_size[1] / resolution))
    layer_names = ["drivable_area", "ped_crossing", "road_divider", "lane_divider"]
    positive_pixels = np.zeros(len(MAP_CLASSES), dtype=np.int64)
    nonempty_samples = np.zeros(len(MAP_CLASSES), dtype=np.int64)
    manifest: List[dict] = []

    print(f"Generating BEV labels (rasterizing HD Map) for {len(frames)} frames...", flush=True)
    for idx, frame in enumerate(frames):
        if idx % 10 == 0 or idx == len(frames) - 1:
            print(f"Rasterizing map mask {idx + 1}/{len(frames)}...", flush=True)
        ego_rotation = quaternion_matrix(frame["ego_rotation"])
        lidar_rotation = quaternion_matrix(frame["lidar_rotation"])
        global_rotation = ego_rotation @ lidar_rotation
        global_translation = ego_rotation @ frame["lidar_translation"] + frame["ego_translation"]
        yaw = math.degrees(math.atan2(global_rotation[1, 0], global_rotation[0, 0]))
        masks = nusc_map.get_map_mask(
            patch_box=(global_translation[0], global_translation[1], *patch_size),
            patch_angle=yaw,
            layer_names=layer_names,
            canvas_size=canvas_size,
        ).astype(bool)
        # Match LoadBEVSegmentation exactly.
        masks = masks.transpose(0, 2, 1)
        labels = np.stack(
            [masks[0], masks[1], np.logical_or(masks[2], masks[3])], axis=0
        ).astype(np.uint8)
        pixel_counts = labels.reshape(len(MAP_CLASSES), -1).sum(axis=1).astype(np.int64)
        positive_pixels += pixel_counts
        nonempty_samples += pixel_counts > 0
        label_path = labels_dir / f"{frame['sample_token']}.npz"
        np.savez_compressed(
            label_path,
            gt_masks_bev=labels,
            classes=np.asarray(MAP_CLASSES),
            sample_token=frame["sample_token"],
            timestamp=np.int64(frame["timestamp_us"]),
        )
        manifest.append(
            {
                "sample_token": frame["sample_token"],
                "split": frame["split"],
                "timestamp": frame["timestamp_us"],
                "path": str(label_path.relative_to(output_root)),
                "positive_pixels": {
                    name: int(value) for name, value in zip(MAP_CLASSES, pixel_counts)
                },
            }
        )

    json_dump(labels_dir / "manifest.json", manifest)
    return {
        "shape": [len(MAP_CLASSES), *canvas_size],
        "resolution_m_per_pixel": resolution,
        "patch_m": list(patch_size),
        "positive_pixels": {
            name: int(value) for name, value in zip(MAP_CLASSES, positive_pixels)
        },
        "nonempty_samples": {
            name: int(value) for name, value in zip(MAP_CLASSES, nonempty_samples)
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--vinfast-root",
        type=Path,
        default=Path("VinFast - data sample"),
        help="Raw VinFast sample directory",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/vinfast-nuscenes"),
        help="nuScenes-compatible output root",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--max-sweeps", type=int, default=9)
    parser.add_argument("--max-camera-lidar-delta-ms", type=float, default=50.0)
    parser.add_argument("--max-lidar-merge-delta-ms", type=float, default=10.0)
    parser.add_argument("--label-resolution", type=float, default=0.5)
    parser.add_argument(
        "--overwrite-lidar",
        action="store_true",
        help="Regenerate existing samples/LIDAR_TOP .bin files instead of reusing them.",
    )
    parser.add_argument(
        "--undistort-images",
        action="store_true",
        help="Write undistorted JPEGs; otherwise create space-saving relative symlinks",
    )
    parser.add_argument(
        "--allow-empty-labels",
        action="store_true",
        help=(
            "Write the dataset even when all BEV map labels are empty. "
            "This is only useful for debugging conversion artifacts, not for training/evaluation."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not 0.0 < args.train_ratio < 1.0:
        raise ValueError("--train-ratio must be between 0 and 1")
    summary = prepare(args)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
