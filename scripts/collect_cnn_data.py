"""현재 장면의 RGB crop을 장면별 분할로 수집한다. 정답 ID는 라벨 확인에만 사용한다."""
from terminal_ko import KoreanArgumentParser, failure_name
from collections import Counter
from datetime import datetime
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import random
import shutil

import numpy as np
from PIL import Image
import pybullet as p
from tqdm.auto import tqdm

from multi_object_scene import LAYOUT, build_scene, settle_scene
from rgbd_camera import capture, components, depth_metres, rgb_crop, world_points

CLASSES = ["cuboid", "cylinder"]


def sample_scene(seed, minimum=0.03, maximum=0.064, min_height=0.03, max_height=0.07):
    # 입력 단위는 m다. 높이와 손가락으로 잡을 가로 폭은 따로 설정한다.
    # 6.5cm는 이론 상한이므로 영상 추정 오차에 1mm 여유를 남긴다.
    if (any(not math.isfinite(n) for n in (minimum, maximum, min_height, max_height))
            or not 0 < minimum <= maximum <= 0.065
            or not 0.03 <= min_height <= max_height <= 0.07):
        raise ValueError("폭은0cm 초과~6.5cm, 높이는3~7cm 범위에서 최솟값이 최댓값 이하가 되도록 지정하세요")
    rng = random.Random(seed)
    count = rng.randint(1, 5)
    shapes = ([rng.choice(CLASSES)] if count == 1 else CLASSES + [rng.choice(CLASSES) for _ in range(count - 2)])
    rng.shuffle(shapes)
    objects = []
    for shape in shapes:
        size = [rng.uniform(minimum, maximum) for _ in range(2 if shape == "cuboid" else 1)] + [rng.uniform(min_height, max_height)]
        yaw = rng.uniform(-math.pi, math.pi)
        # 회전한 물체 전체를 구역 안에 넣고 이웃 물체와 손가락 여유를 확보한다.
        if shape == "cuboid":
            half = [(abs(math.cos(yaw)) * size[0] + abs(math.sin(yaw)) * size[1]) / 2,
                    (abs(math.sin(yaw)) * size[0] + abs(math.cos(yaw)) * size[1]) / 2]
        else:
            half = [(size[0] / 2 + 0.001) * (abs(math.cos(yaw)) + abs(math.sin(yaw)))] * 2
        limits = [(LAYOUT["region"][k][0] + half[a] + 0.003, LAYOUT["region"][k][1] - half[a] - 0.003)
                  for a, k in enumerate(("x", "y"))]
        for _ in range(100):
            xy = [rng.uniform(low, high) for low, high in limits]
            if all(any(abs(xy[a] - other["xy"][a]) >= half[a] + other["half_xy"][a] + LAYOUT["gap_m"] + 0.002
                       for a in (0, 1)) for other in objects):
                objects.append({"shape": shape, "size_m": size, "xy": xy, "yaw": yaw, "half_xy": half})
                break
        else:
            raise ValueError("placement_exhausted")
    return objects


def labelled_crops(scene):
    camera = scene["camera"]
    observation = capture(camera)
    points = world_points(observation)
    # 실행 때와 같은 깊이 차이·작업 구역으로 물체를 찾는다. 정답 ID로 자르지 않는다.
    mask = depth_metres(scene["empty"]["depth"], camera) - depth_metres(observation["depth"], camera) > camera["depth_difference_m"]
    for axis, (low, high) in enumerate(camera["roi_xy_m"]):
        mask &= (points[..., axis] > low) & (points[..., axis] < high)
    mask &= (points[..., 2] > camera["table_top_m"] + 0.008) & (observation["depth"] < 1)
    groups = components(mask)
    # 별도 ID 영상은 학습 라벨 확인 전용이다. 좌표 추정과 CNN 입력에는 전달하지 않는다.
    frame = p.getCameraImage(camera["width"], camera["height"], observation["view"], observation["projection"],
                             renderer=p.ER_TINY_RENDERER, flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX)
    annotation = np.asarray(frame[4], dtype=np.int64).reshape(mask.shape)
    annotation = np.where(annotation < 0, -1, annotation & ((1 << 24) - 1))
    truth = {obj["body"]: obj for obj in scene["objects"]}
    crops, seen = [], set()
    for group in groups:
        pixels = np.asarray(group)
        region = np.zeros_like(mask)
        region[pixels[:, 0], pixels[:, 1]] = True
        ids, counts = np.unique(annotation[region], return_counts=True)
        winner = int(ids[counts.argmax()])
        purity = float(counts.max() / counts.sum())
        if winner not in truth or purity < 0.99 or winner in seen:
            raise ValueError("ambiguous_annotation")
        # 구역 경계에서 잘리거나 두 물체가 붙은 영역은 학습 이미지로 사용하지 않는다.
        surface = points[region]
        if any(surface[:, a].min() < lo + 0.002 or surface[:, a].max() > hi - 0.002
               for a, (lo, hi) in enumerate(camera["roi_xy_m"])):
            raise ValueError("cropped_at_roi_edge")
        seen.add(winner)
        crops.append((rgb_crop(observation["rgb"], region, camera["crop_size"]), truth[winner], purity))
    if seen != set(truth):
        raise ValueError("hidden_or_merged_object")
    return crops, observation


def collect(output, scenes=2000, seed=2026, minimum=0.03, maximum=0.064,
            min_height=0.03, max_height=0.07, show_progress=True):
    sample_scene(seed, minimum, maximum, min_height, max_height)
    if type(scenes) is not int or scenes < 3:
        raise ValueError("분할을 위해 장면을 최소 3개 지정하세요")
    output = Path(output)
    if output.exists() or output.with_suffix(".zip").exists():
        raise FileExistsError("기존 데이터를 덮어쓰지 않습니다. 새 출력 폴더를 지정하세요")
    output.mkdir(parents=True)
    # 장면을 먼저 분할한다. 같은 장면의 물체들은 다른 분할에 섞지 않는다.
    n_val = max(1, int(scenes * 0.15))
    n_test = max(1, int(scenes * 0.15))
    splits = ["train"] * (scenes - n_val - n_test) + ["val"] * n_val + ["test"] * n_test
    random.Random(seed).shuffle(splits)
    for split in ("train", "val", "test"):
        for label in CLASSES:
            (output / split / label).mkdir(parents=True)
    config = {"schema_version": 1, "classes": CLASSES, "scenes": scenes, "seed": seed,
              "width_range_m": [minimum, maximum], "height_range_m": [min_height, max_height],
              "split_method": "scene_group_70_15_15", "crop_size": 64,
              "preprocessing": {"rgb": True, "resize": "PIL_BILINEAR", "crop_total_margin_px": 12,
                                "tensor_scale": "uint8/255", "normalize_mean": [0.5] * 3, "normalize_std": [0.5] * 3},
              "versions": {name: importlib.metadata.version(name) for name in ("pybullet", "numpy", "Pillow", "tqdm")},
              "source_sha256": {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                                for name in ("collect_cnn_data.py", "multi_object_scene.py", "rgbd_camera.py")},
              "status": "collecting", "size_transport_verified": False}
    (output / "config.json").write_text(json.dumps(config, indent=2))
    counts = Counter()
    client = p.connect(p.DIRECT)
    try:
        with (output / "manifest.jsonl").open("w") as manifest, (output / "rejected_scenes.jsonl").open("w") as rejected:
            for index, split in enumerate(tqdm(splits, desc="장면 수집", disable=not show_progress,
                                               bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}장면 [경과 {elapsed}, 남음 {remaining}]")):
                for attempt in range(30):
                    scene_seed = seed + index * 1000 + attempt
                    try:
                        spawns = sample_scene(scene_seed, minimum, maximum, min_height, max_height)
                        scene = build_scene(spawns, LAYOUT)
                        settled = settle_scene(scene)
                        if not settled["settled"]:
                            raise ValueError(settled["failure_reason"])
                        crops, observation = labelled_crops(scene)
                        # 각 분할의 첫 장면은 두 클래스가 들어 있게 해 아주 작은 데이터도 평가 가능하게 한다.
                        if not all(counts[(split, label)] for label in CLASSES) and {obj["shape"] for _, obj, _ in crops} != set(CLASSES):
                            raise ValueError("need_both_classes_for_split")
                        break
                    except ValueError as error:
                        rejected.write(json.dumps({"scene_id": index, "seed": scene_seed, "reason": str(error)}) + "\n")
                else:
                    raise RuntimeError(f"장면 {index}: 30번 시도해도 수집할 수 없습니다")
                config["camera"] = scene["camera"]
                # 전체 RGB 몇 장도 남겨 실제 구도와 crop을 사람이 확인할 수 있게 한다.
                if index < 12:
                    (output / "preview").mkdir(exist_ok=True)
                    Image.fromarray(observation["rgb"]).save(output / "preview" / f"scene_{index:05d}.png")
                for obj_index, (image, obj, purity) in enumerate(crops):
                    path = Path(split) / obj["shape"] / f"scene_{index:05d}_{obj_index}.png"
                    image.save(output / path)
                    row = {"scene_id": index, "seed": scene_seed, "split": split, "label": obj["shape"],
                           "path": path.as_posix(), "size_m": obj["size_m"], "spawn_xy_m": obj["xy"],
                           "yaw_rad": obj["yaw"], "objects_in_scene": len(crops), "annotation_purity": purity}
                    manifest.write(json.dumps(row) + "\n")
                    counts[(split, obj["shape"])] += 1
                manifest.flush()
        config.update(status="complete", image_counts={f"{s}/{c}": counts[(s, c)] for s in ("train", "val", "test") for c in CLASSES})
        config["manifest_sha256"] = hashlib.sha256((output / "manifest.jsonl").read_bytes()).hexdigest()
        (output / "config.json").write_text(json.dumps(config, indent=2))
        shutil.make_archive(str(output), "zip", root_dir=output)
        return config
    except BaseException:
        config["status"] = "incomplete"
        (output / "config.json").write_text(json.dumps(config, indent=2))
        raise
    finally:
        p.disconnect(client)


def main():
    parser = KoreanArgumentParser(description=__doc__)
    parser.add_argument("--scenes", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--min-width-cm", type=float, default=3)
    parser.add_argument("--max-width-cm", type=float, default=6.4)
    parser.add_argument("--min-height-cm", type=float, default=3)
    parser.add_argument("--max-height-cm", type=float, default=7)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output = args.output_dir or Path("data") / f"shape_cnn_{datetime.now():%Y%m%dT%H%M%S_%f}"
    try:
        result = collect(output, args.scenes, args.seed, args.min_width_cm / 100, args.max_width_cm / 100,
                         args.min_height_cm / 100, args.max_height_cm / 100)
    except (ValueError, RuntimeError, FileExistsError) as error:
        parser.error(failure_name(str(error)))
    print(f"데이터 폴더: {output}\n압축 파일: {output.with_suffix('.zip')}")
    for split, label in [('train', '학습'), ('val', '검증'), ('test', '테스트')]:
        counts = result['image_counts']
        print(f"{label} 사진 수: 직육면체 {counts[f'{split}/cuboid']}장, 원기둥 {counts[f'{split}/cylinder']}장")


if __name__ == "__main__":
    main()
