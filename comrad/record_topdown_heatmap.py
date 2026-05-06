import os
import sys
from dataclasses import dataclass
from os.path import join
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import imageio
import numpy as np
import torch

from sample_factory.algo.learning.learner import Learner
from sample_factory.algo.sampling.batched_sampling import preprocess_actions
from sample_factory.algo.utils.action_distributions import argmax_actions
from sample_factory.algo.utils.env_info import extract_env_info
from sample_factory.algo.utils.make_env import make_env_func_batched
from sample_factory.algo.utils.misc import ExperimentStatus
from sample_factory.algo.utils.rl_utils import make_dones, prepare_and_normalize_obs
from sample_factory.algo.utils.tensor_utils import unsqueeze_tensor
from sample_factory.cfg.arguments import load_from_checkpoint, parse_full_cfg, parse_sf_args
from sample_factory.model.actor_critic import create_actor_critic
from sample_factory.model.model_utils import get_rnn_size
from sample_factory.utils.attr_dict import AttrDict
from sample_factory.utils.typing import Config, StatusCode
from sample_factory.utils.utils import log

from comrad.envs.doom_params import add_doom_env_args, add_doom_env_eval_args, add_wandb_args, doom_override_defaults
from comrad.train import register_model_factory, register_vizdoom_components
from comrad.utils.doom_utils import DOOM_ENVS


AUTOMAP_CAPTURE_RESOLUTION = "1600x1200"
AUTOMAP_BACKGROUND_RGB = np.array([239, 239, 239], dtype=np.uint8)
AUTOMAP_WALL_RGB = np.array([138, 132, 120], dtype=np.uint8)
AUTOMAP_PLAYER0_RGB = np.array([255, 0, 255], dtype=np.uint8)
AUTOMAP_PLAYER1_RGB = np.array([95, 207, 87], dtype=np.uint8)
HEATMAP_COLORS = (
    np.array([255, 96, 84], dtype=np.float32),
    np.array([66, 156, 255], dtype=np.float32),
)
AMMO_CARRIER_PLATFORM_WORLD_POINTS = (
    (-200.0, 200.0),
    (200.0, 200.0),
    (-200.0, -200.0),
    (200.0, -200.0),
)
AMMO_CARRIER_DEPOT_CENTER_WORLD_POINT = (960.0, 0.0)


@dataclass
class EpisodeTrace:
    positions: Dict[str, List[Optional[Tuple[float, float]]]]
    rewards: np.ndarray
    env_steps: int


@dataclass
class MarkerDetection:
    centroid: Tuple[float, float]
    mask: np.ndarray


@dataclass
class OpenComponent:
    area: int
    centroid: Tuple[float, float]
    mask: np.ndarray


@dataclass
class AutomapCalibration:
    background_rgb: np.ndarray
    world_to_map: np.ndarray
    walkable_mask: Optional[np.ndarray] = None


def add_topdown_args(parser) -> None:
    """
    Example: uv run python -m comrad.record_topdown_heatmap --env=ammo_carrier --train_dir=train_dir/.../ammo_carrier --experiment=IPPO_env_ammo_carrier --output_dir=results/videos/test --device=cpu --overwrite
    """
    parser.add_argument(
        "--resolution",
        default=AUTOMAP_CAPTURE_RESOLUTION,
        type=str,
        help="Output resolution in WIDTHxHEIGHT format. Must keep the static automap aspect ratio.",
    )
    parser.add_argument("--video_fps", default=35, type=int, help="Output video FPS")
    parser.add_argument("--output_dir", default="results/videos", type=str, help="Output directory for mp4/png files")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    parser.add_argument("--heatmap_radius", default=64, type=int, help="Heatmap stamp radius in pixels")
    parser.add_argument("--heatmap_alpha", default=0.72, type=float, help="Max alpha for the heatmap overlay")
    parser.add_argument("--line_thickness", default=4, type=int, help="Trajectory line thickness in pixels")
    parser.add_argument("--marker_radius", default=8, type=int, help="Current-position marker radius in pixels")
    parser.add_argument(
        "--density_grid_resolution",
        default="200x150",
        type=str,
        help="Grid resolution for the automap-aligned density heatmap. Must keep the automap aspect ratio.",
    )
    parser.add_argument(
        "--density_heatmap_alpha",
        default=0.82,
        type=float,
        help="Max alpha for the grid-density heatmap overlay",
    )
    parser.set_defaults(
        max_num_episodes=1,
        max_num_frames=None,
        eval_deterministic=True,
        save_video=False,
        load_checkpoint_kind="best",
    )


def parse_topdown_args(argv=None) -> Config:
    parser, _ = parse_sf_args(argv=argv, evaluation=True)
    add_doom_env_args(parser)
    add_doom_env_eval_args(parser)
    add_wandb_args(parser)
    add_topdown_args(parser)
    doom_override_defaults(parser)
    return parse_full_cfg(parser, argv)


def parse_resolution(resolution: str) -> Tuple[int, int]:
    try:
        width, height = map(int, resolution.lower().split("x"))
    except ValueError as exc:
        raise ValueError(f"Expected resolution like 1600x1200, got {resolution!r}") from exc
    if width <= 0 or height <= 0:
        raise ValueError(f"Resolution must be positive, got {resolution!r}")
    return width, height


def assert_matching_automap_aspect(resolution: str) -> None:
    out_w, out_h = parse_resolution(resolution)
    map_w, map_h = parse_resolution(AUTOMAP_CAPTURE_RESOLUTION)
    if out_w * map_h != out_h * map_w:
        raise ValueError(
            f"{resolution} does not match automap aspect ratio {AUTOMAP_CAPTURE_RESOLUTION}; "
            "use a matching 4:3 resolution to avoid heatmap/map distortion"
        )


def save_video(frames: Sequence[np.ndarray], file_path: str, fps: int) -> None:
    writer = imageio.get_writer(file_path, fps=fps, quality=8, macro_block_size=1)
    for frame in frames:
        writer.append_data(frame)
    writer.close()
    log.info("Saved video %s (%d frames at %d fps)", file_path, len(frames), fps)


def save_image(frame: np.ndarray, file_path: str) -> None:
    cv2.imwrite(file_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    log.info("Saved image %s", file_path)


def algo_label(cfg: Config) -> str:
    algo = str(cfg.algo).upper()
    if algo == "QPLEX":
        mixer = str(getattr(cfg, "mixer", "")).lower()
        if mixer:
            return f"{algo}_{mixer}"
    return algo


def output_stem(cfg: Config, episode_idx: int) -> str:
    return f"{cfg.env}_{algo_label(cfg)}_topdown_heatmap_best_ep{episode_idx:02d}_{cfg.resolution}"


def density_output_stem(cfg: Config, episode_idx: int) -> str:
    return f"{output_stem(cfg, episode_idx)}_grid_density"


def make_env(cfg: Config):
    return make_env_func_batched(
        cfg,
        env_config=AttrDict(worker_index=0, vector_index=0, env_id=0),
        render_mode=None,
    )


def reshape_deterministic_actions(sampled_actions, deterministic_actions):
    if type(sampled_actions) is not type(deterministic_actions):
        return deterministic_actions

    sampled_shape = getattr(sampled_actions, "shape", None)
    deterministic_shape = getattr(deterministic_actions, "shape", None)
    if sampled_shape is None or deterministic_shape is None:
        return deterministic_actions
    if sampled_shape == deterministic_shape:
        return deterministic_actions

    sampled_numel = int(np.prod(sampled_shape))
    deterministic_numel = int(np.prod(deterministic_shape))
    if sampled_numel != deterministic_numel:
        return deterministic_actions

    if len(sampled_shape) == 2 and len(deterministic_shape) == 2 and deterministic_shape[0] == 1:
        num_agents, num_heads = sampled_shape
        head_major = deterministic_actions.reshape(num_heads, num_agents)
        if isinstance(head_major, torch.Tensor):
            return head_major.transpose(0, 1)
        return np.swapaxes(head_major, 0, 1)

    return deterministic_actions.reshape(sampled_shape)


def load_policy(cfg: Config, env, device: torch.device):
    actor_critic = create_actor_critic(cfg, env.observation_space, env.action_space)
    actor_critic.eval()
    actor_critic.model_to_device(device)

    policy_id = cfg.policy_index
    name_prefix = dict(latest="checkpoint", best="best")[cfg.load_checkpoint_kind]
    checkpoints = Learner.get_checkpoints(Learner.checkpoint_dir(cfg, policy_id), f"{name_prefix}_*")
    checkpoint_dict = Learner.load_checkpoint(checkpoints, device)
    if checkpoint_dict is None:
        raise RuntimeError("Could not load checkpoint")
    actor_critic.load_state_dict(checkpoint_dict["model"])
    return actor_critic


def scenario_cfg_path(env_name: str) -> str:
    for spec in DOOM_ENVS:
        if spec.name == env_name:
            return os.path.abspath(join(os.path.dirname(__file__), "scenarios", spec.env_spec_file))
    raise KeyError(f"Unknown Doom env {env_name!r}")


def _position_from_info(info: Optional[dict]) -> Optional[Tuple[float, float]]:
    if not isinstance(info, dict):
        return None

    pos = info.get("pos")
    if isinstance(pos, dict):
        x = pos.get("agent_x")
        y = pos.get("agent_y")
    else:
        x = info.get("POSITION_X")
        y = info.get("POSITION_Y")

    if x is None or y is None:
        return None

    x = float(x)
    y = float(y)
    if not np.isfinite(x) or not np.isfinite(y):
        return None
    return x, y


def extract_positions_from_infos(
    infos: Sequence[Optional[dict]],
    last_positions: Optional[Dict[str, Optional[Tuple[float, float]]]] = None,
) -> Dict[str, Optional[Tuple[float, float]]]:
    positions = {}
    last_positions = last_positions or {}
    for agent_idx, info in enumerate(infos):
        name = f"agent{agent_idx}"
        positions[name] = _position_from_info(info) or last_positions.get(name)
    return positions


def compute_world_bounds(
    positions: Dict[str, Sequence[Optional[Tuple[float, float]]]],
    padding_ratio: float = 0.08,
) -> Tuple[float, float, float, float]:
    valid_points = [point for path in positions.values() for point in path if point is not None]
    if not valid_points:
        raise RuntimeError("No telemetry positions were captured")

    xs = np.asarray([point[0] for point in valid_points], dtype=np.float32)
    ys = np.asarray([point[1] for point in valid_points], dtype=np.float32)
    min_x, max_x = float(xs.min()), float(xs.max())
    min_y, max_y = float(ys.min()), float(ys.max())
    span = max(max_x - min_x, max_y - min_y, 1.0)
    pad = span * padding_ratio
    return min_x - pad, min_y - pad, max_x + pad, max_y + pad


def world_to_canvas(
    position: Optional[Tuple[float, float]],
    bounds: Tuple[float, float, float, float],
    width: int,
    height: int,
) -> Optional[Tuple[int, int]]:
    if position is None:
        return None

    min_x, min_y, max_x, max_y = bounds
    px = int(round((position[0] - min_x) * width / max(max_x - min_x, 1e-6)))
    py = int(round(height - 1 - (position[1] - min_y) * height / max(max_y - min_y, 1e-6)))
    return int(np.clip(px, 0, width - 1)), int(np.clip(py, 0, height - 1))


def normalize_map_buffer(buffer: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if buffer is None:
        return None
    image = np.asarray(buffer)
    if image.ndim != 3:
        return None
    if image.shape[0] == 3 and image.shape[-1] != 3:
        image = np.transpose(image, (1, 2, 0))
    return np.ascontiguousarray(image)


def _configure_static_automap(game) -> None:
    import vizdoom as vzd

    game.set_automap_buffer_enabled(True)
    game.set_automap_mode(vzd.AutomapMode.OBJECTS_WITH_SIZE)
    game.set_automap_rotate(False)
    game.set_automap_render_textures(False)
    game.add_game_args("+viz_am_center 1")
    game.add_game_args("+am_backcolor f1eee6")
    game.add_game_args("+am_tswallcolor 8a8478")
    game.add_game_args("+am_yourcolor ff00ff")
    game.add_game_args("+am_cheat 0")
    game.add_game_args("+am_thingcolor 5b8cff")
    game.add_game_args("+am_thingcolor_item 44c989")


def _component_detections(mask: np.ndarray) -> List[MarkerDetection]:
    num_components, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    detections: List[MarkerDetection] = []
    for idx in range(1, num_components):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        width = int(stats[idx, cv2.CC_STAT_WIDTH])
        height = int(stats[idx, cv2.CC_STAT_HEIGHT])
        if area < 50 or area > 2000:
            continue
        if min(width, height) < 6 or width > 80 or height > 80:
            continue
        detections.append(
            MarkerDetection(
                centroid=(float(centroids[idx][0]), float(centroids[idx][1])),
                mask=labels == idx,
            )
        )
    return detections


def _closest_color_mask(background_rgb: np.ndarray, target_rgb: np.ndarray, tolerance: int = 48) -> np.ndarray:
    distance = np.abs(background_rgb.astype(np.int16) - target_rgb.astype(np.int16)).max(axis=2)
    return distance <= tolerance


def detect_start_markers(background_rgb: np.ndarray) -> List[MarkerDetection]:
    detections: List[MarkerDetection] = []

    for target_rgb in (AUTOMAP_PLAYER0_RGB, AUTOMAP_PLAYER1_RGB):
        candidates = _component_detections(_closest_color_mask(background_rgb, target_rgb))
        if not candidates:
            continue
        candidates.sort(key=lambda det: int(det.mask.sum()), reverse=True)
        detections.append(candidates[0])

    if len(detections) >= 2:
        detections.sort(key=lambda det: det.centroid[0])
        return detections[:2]

    chroma = background_rgb.max(axis=2) - background_rgb.min(axis=2)
    brightness = background_rgb.mean(axis=2)
    fallback = _component_detections((chroma >= 50) & (brightness >= 80))
    fallback.sort(key=lambda det: det.centroid[0])
    return fallback[:2]


def detect_start_marker_candidates(background_rgb: np.ndarray, per_color_limit: int = 6) -> List[MarkerDetection]:
    detections: List[MarkerDetection] = []
    for target_rgb in (AUTOMAP_PLAYER0_RGB, AUTOMAP_PLAYER1_RGB):
        candidates = _component_detections(_closest_color_mask(background_rgb, target_rgb))
        candidates.sort(key=lambda det: int(det.mask.sum()), reverse=True)
        detections.extend(candidates[:per_color_limit])

    if not detections:
        chroma = background_rgb.max(axis=2) - background_rgb.min(axis=2)
        brightness = background_rgb.mean(axis=2)
        detections = _component_detections((chroma >= 50) & (brightness >= 80))

    detections.sort(key=lambda det: int(det.mask.sum()), reverse=True)
    unique: List[MarkerDetection] = []
    for detection in detections:
        if all(np.hypot(detection.centroid[0] - other.centroid[0], detection.centroid[1] - other.centroid[1]) >= 14.0 for other in unique):
            unique.append(detection)

    unique.sort(key=lambda det: det.centroid[0])
    return unique


def scrub_detected_markers(background_rgb: np.ndarray, detections: Sequence[MarkerDetection]) -> np.ndarray:
    def background_fill_color(image: np.ndarray) -> np.ndarray:
        return np.median(
            np.concatenate(
                [
                    image[0].reshape(-1, 3),
                    image[-1].reshape(-1, 3),
                    image[:, 0].reshape(-1, 3),
                    image[:, -1].reshape(-1, 3),
                ],
                axis=0,
            ),
            axis=0,
        ).astype(np.uint8)

    if not detections:
        return background_rgb

    scrubbed = background_rgb.copy()
    background_color = background_fill_color(scrubbed)

    for detection in detections:
        dilated = cv2.dilate(
            detection.mask.astype(np.uint8),
            np.ones((5, 5), dtype=np.uint8),
            iterations=1,
        ).astype(bool)
        scrubbed[dilated] = background_color

    return scrubbed


def scrub_colored_automap_artifacts(
    background_rgb: np.ndarray,
    chroma_threshold: int = 60,
    max_channel_threshold: int = 180,
) -> np.ndarray:
    scrubbed = background_rgb.copy()
    chroma = scrubbed.max(axis=2) - scrubbed.min(axis=2)
    max_channel = scrubbed.max(axis=2)
    artifact_mask = (chroma >= chroma_threshold) & (max_channel >= max_channel_threshold)
    if not np.any(artifact_mask):
        return scrubbed

    background_color = np.median(
        np.concatenate(
            [
                scrubbed[0].reshape(-1, 3),
                scrubbed[-1].reshape(-1, 3),
                scrubbed[:, 0].reshape(-1, 3),
                scrubbed[:, -1].reshape(-1, 3),
            ],
            axis=0,
        ),
        axis=0,
    ).astype(np.uint8)
    dilated = cv2.dilate(artifact_mask.astype(np.uint8), np.ones((5, 5), dtype=np.uint8), iterations=1).astype(bool)
    scrubbed[dilated] = background_color
    return scrubbed


def build_walkable_mask(background_rgb: np.ndarray, wall_threshold: int = 18, wall_padding: int = 2) -> np.ndarray:
    background_delta = np.abs(background_rgb.astype(np.int16) - AUTOMAP_BACKGROUND_RGB.astype(np.int16)).max(axis=2)
    wall_mask = background_delta >= wall_threshold
    if wall_padding > 0:
        wall_mask = cv2.dilate(
            wall_mask.astype(np.uint8),
            np.ones((wall_padding * 2 + 1, wall_padding * 2 + 1), dtype=np.uint8),
            iterations=1,
        ).astype(bool)

    open_pixels = (~wall_mask).astype(np.uint8)
    num_components, labels = cv2.connectedComponents(open_pixels, connectivity=4)
    if num_components <= 1:
        raise RuntimeError("Could not extract walkable automap region")

    border_labels = np.unique(
        np.concatenate(
            [
                labels[0, :],
                labels[-1, :],
                labels[:, 0],
                labels[:, -1],
            ]
        )
    )
    outside_mask = np.isin(labels, border_labels) & (labels != 0)
    walkable_mask = (labels != 0) & (~outside_mask)
    if not np.any(walkable_mask):
        raise RuntimeError("Static automap produced no enclosed walkable region")
    return walkable_mask


def detect_enclosed_open_components(
    background_rgb: np.ndarray,
    wall_threshold: int = 10,
    wall_padding: int = 1,
    min_area: int = 32,
) -> List[OpenComponent]:
    background_delta = np.abs(background_rgb.astype(np.int16) - AUTOMAP_BACKGROUND_RGB.astype(np.int16)).max(axis=2)
    wall_mask = background_delta >= wall_threshold
    if wall_padding > 0:
        kernel = np.ones((wall_padding * 2 + 1, wall_padding * 2 + 1), dtype=np.uint8)
        wall_mask = cv2.dilate(wall_mask.astype(np.uint8), kernel, iterations=1).astype(bool)

    open_pixels = (~wall_mask).astype(np.uint8)
    num_components, labels, stats, centroids = cv2.connectedComponentsWithStats(open_pixels, connectivity=4)
    components: List[OpenComponent] = []
    height, width = background_rgb.shape[:2]
    for idx in range(1, num_components):
        area = int(stats[idx, cv2.CC_STAT_AREA])
        if area < min_area:
            continue

        x, y, w, h = (int(stats[idx, axis]) for axis in range(4))
        touches_border = x == 0 or y == 0 or x + w == width or y + h == height
        if touches_border:
            continue

        components.append(
            OpenComponent(
                area=area,
                centroid=(float(centroids[idx][0]), float(centroids[idx][1])),
                mask=labels == idx,
            )
        )

    components.sort(key=lambda component: component.area, reverse=True)
    return components


def walkable_distance_map(walkable_mask: np.ndarray) -> np.ndarray:
    return cv2.distanceTransform(walkable_mask.astype(np.uint8), cv2.DIST_L2, 5)


def capture_static_automap(env_name: str) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
    import vizdoom as vzd

    config_path = scenario_cfg_path(env_name)
    game = vzd.DoomGame()
    game.load_config(config_path)
    game.set_render_hud(False)
    width, height = parse_resolution(AUTOMAP_CAPTURE_RESOLUTION)
    game.set_screen_resolution(getattr(vzd.ScreenResolution, f"RES_{width}X{height}"))
    game.set_screen_format(vzd.ScreenFormat.RGB24)
    _configure_static_automap(game)
    game.set_window_visible(False)

    try:
        game.init()
        game.new_episode()
        state = game.get_state()
        if state is None or state.automap_buffer is None:
            raise RuntimeError(f"Could not capture static automap for {env_name}")

        image = normalize_map_buffer(state.automap_buffer)
        detections = detect_start_marker_candidates(image)
        if len(detections) < 2:
            raise RuntimeError(f"Could not detect both start markers on static automap for {env_name}")

        scrubbed = scrub_colored_automap_artifacts(scrub_detected_markers(image, detections))
        marker_points = [detection.centroid for detection in detections]
        return scrubbed, marker_points
    finally:
        game.close()


def fit_spawn_alignment(
    world_points: Sequence[Tuple[float, float]],
    map_points: Sequence[Tuple[float, float]],
) -> np.ndarray:
    if len(world_points) < 2 or len(map_points) < 2:
        raise ValueError("Need at least two world points and two map points for alignment")

    world_sorted = sorted(world_points, key=lambda point: point[0])
    map_sorted = sorted(map_points, key=lambda point: point[0])
    (wx0, wy0), (wx1, wy1) = world_sorted[:2]
    (px0, py0), (px1, py1) = map_sorted[:2]

    if abs(wx1 - wx0) < 1e-6 or abs(wy1 - wy0) < 1e-6:
        raise ValueError("Spawn positions are degenerate for transform fitting")

    scale_x = (px1 - px0) / (wx1 - wx0)
    scale_y = (py1 - py0) / (wy1 - wy0)
    offset_x = px0 - scale_x * wx0
    offset_y = py0 - scale_y * wy0
    return np.array([[scale_x, 0.0, offset_x], [0.0, scale_y, offset_y]], dtype=np.float32)


def fit_affine_alignment(
    world_points: Sequence[Tuple[float, float]],
    map_points: Sequence[Tuple[float, float]],
) -> np.ndarray:
    if len(world_points) != len(map_points):
        raise ValueError("world_points and map_points must have matching lengths")
    if len(world_points) < 3:
        raise ValueError("Need at least three correspondences for affine alignment")

    world = np.asarray(world_points, dtype=np.float64)
    target = np.asarray(map_points, dtype=np.float64)
    design = np.column_stack([world[:, 0], world[:, 1], np.ones(len(world), dtype=np.float64)])
    coeff_x, *_ = np.linalg.lstsq(design, target[:, 0], rcond=None)
    coeff_y, *_ = np.linalg.lstsq(design, target[:, 1], rcond=None)
    return np.array(
        [
            [coeff_x[0], coeff_x[1], coeff_x[2]],
            [coeff_y[0], coeff_y[1], coeff_y[2]],
        ],
        dtype=np.float32,
    )


def combined_trace_points(
    positions: Dict[str, Sequence[Optional[Tuple[float, float]]]],
) -> List[Tuple[float, float]]:
    return [point for path in positions.values() for point in path if point is not None]


def project_with_transform_float(
    transform: np.ndarray, position: Optional[Tuple[float, float]]
) -> Optional[Tuple[float, float]]:
    if position is None:
        return None
    vec = np.array([position[0], position[1], 1.0], dtype=np.float32)
    px, py = transform @ vec
    return float(px), float(py)


def project_with_transform(transform: np.ndarray, position: Optional[Tuple[float, float]]) -> Optional[Tuple[int, int]]:
    projected = project_with_transform_float(transform, position)
    if projected is None:
        return None
    px, py = projected
    return int(round(px)), int(round(py))


def walkable_bbox(walkable_mask: np.ndarray) -> Tuple[float, float, float, float]:
    ys, xs = np.nonzero(walkable_mask)
    if xs.size == 0 or ys.size == 0:
        raise RuntimeError("Walkable mask is empty")
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())


def bbox_alignment_candidates(
    positions: Dict[str, Sequence[Optional[Tuple[float, float]]]],
    walkable_mask: np.ndarray,
) -> List[np.ndarray]:
    world_points = combined_trace_points(positions)
    if not world_points:
        raise RuntimeError("No telemetry positions were captured")

    xs = np.asarray([point[0] for point in world_points], dtype=np.float32)
    ys = np.asarray([point[1] for point in world_points], dtype=np.float32)
    min_x, max_x = float(xs.min()), float(xs.max())
    min_y, max_y = float(ys.min()), float(ys.max())
    span_x = max_x - min_x
    span_y = max_y - min_y

    box_min_x, box_min_y, box_max_x, box_max_y = walkable_bbox(walkable_mask)
    box_span_x = box_max_x - box_min_x
    box_span_y = box_max_y - box_min_y
    candidates = []
    for flip_x in (False, True):
        for flip_y in (False, True):
            scale_x = box_span_x / max(span_x, 1e-6)
            scale_y = box_span_y / max(span_y, 1e-6)

            if span_x < 1e-6 and span_y < 1e-6:
                scale_x = scale_y = 1.0
            elif span_x < 1e-6:
                scale_x = abs(scale_y)
            elif span_y < 1e-6:
                scale_y = abs(scale_x)

            if flip_x:
                scale_x = -scale_x
                offset_x = box_max_x - scale_x * min_x if span_x >= 1e-6 else (box_min_x + box_max_x) * 0.5 - scale_x * min_x
            else:
                offset_x = box_min_x - scale_x * min_x if span_x >= 1e-6 else (box_min_x + box_max_x) * 0.5 - scale_x * min_x

            if flip_y:
                scale_y = -scale_y
                offset_y = box_max_y - scale_y * min_y if span_y >= 1e-6 else (box_min_y + box_max_y) * 0.5 - scale_y * min_y
            else:
                offset_y = box_min_y - scale_y * min_y if span_y >= 1e-6 else (box_min_y + box_max_y) * 0.5 - scale_y * min_y

            candidates.append(np.array([[scale_x, 0.0, offset_x], [0.0, scale_y, offset_y]], dtype=np.float32))
    return candidates


def spawn_alignment_candidates(
    positions: Dict[str, Sequence[Optional[Tuple[float, float]]]],
    marker_points: Optional[Sequence[Tuple[float, float]]],
) -> List[np.ndarray]:
    if not marker_points or len(marker_points) < 2:
        return []

    start_points = []
    for agent_name in sorted(positions):
        for point in positions[agent_name]:
            if point is not None:
                start_points.append(point)
                break

    if len(start_points) < 2:
        return []

    candidates = []
    sorted_markers = sorted(marker_points, key=lambda point: point[0])
    for left_idx in range(len(sorted_markers) - 1):
        for right_idx in range(left_idx + 1, len(sorted_markers)):
            try:
                candidates.append(fit_spawn_alignment(start_points[:2], [sorted_markers[left_idx], sorted_markers[right_idx]]))
            except ValueError:
                continue
    return candidates


def _order_quadrant_centroids(centroids: Sequence[Tuple[float, float]]) -> List[Tuple[float, float]]:
    if len(centroids) != 4:
        raise ValueError(f"Expected exactly four centroids, got {len(centroids)}")

    ordered_by_y = sorted(centroids, key=lambda point: (point[1], point[0]))
    top = sorted(ordered_by_y[:2], key=lambda point: point[0])
    bottom = sorted(ordered_by_y[2:], key=lambda point: point[0])
    return [top[0], top[1], bottom[0], bottom[1]]


def detect_ammo_carrier_geometry(background_rgb: np.ndarray) -> Tuple[List[Tuple[float, float]], Tuple[float, float]]:
    components = detect_enclosed_open_components(background_rgb, wall_threshold=10, wall_padding=1, min_area=64)
    height, width = background_rgb.shape[:2]
    image_area = float(height * width)
    min_platform_area = image_area * 0.002
    max_platform_area = image_area * 0.02
    min_room_area = image_area * 0.02

    platform_candidates = [
        component
        for component in components
        if min_platform_area <= component.area <= max_platform_area and component.centroid[0] < width * 0.5
    ]
    if len(platform_candidates) < 4:
        raise RuntimeError(
            f"Could not find the four ammo_carrier hub platforms on the static automap; got {len(platform_candidates)} candidates"
        )

    platform_centroids = _order_quadrant_centroids([component.centroid for component in platform_candidates[:4]])

    depot_candidates = [
        component
        for component in components
        if component.area >= min_room_area and component.centroid[0] > width * 0.6 and component.centroid[1] < height * 0.75
    ]
    if not depot_candidates:
        raise RuntimeError("Could not find the ammo_carrier depot room on the static automap")

    depot_component = max(depot_candidates, key=lambda component: component.centroid[0])
    return platform_centroids, depot_component.centroid


def calibrate_ammo_carrier_to_automap(background_rgb: np.ndarray) -> AutomapCalibration:
    platform_centroids, depot_centroid = detect_ammo_carrier_geometry(background_rgb)
    world_points = list(AMMO_CARRIER_PLATFORM_WORLD_POINTS) + [AMMO_CARRIER_DEPOT_CENTER_WORLD_POINT]
    map_points = list(platform_centroids) + [depot_centroid]
    transform = fit_affine_alignment(world_points, map_points)
    return AutomapCalibration(background_rgb=background_rgb, world_to_map=transform)


def sample_projected_path_points(
    transform: np.ndarray,
    positions: Dict[str, Sequence[Optional[Tuple[float, float]]]],
    spacing_px: float = 3.0,
) -> np.ndarray:
    samples: List[Tuple[float, float]] = []
    for path in positions.values():
        previous = None
        for point in path:
            current = project_with_transform_float(transform, point)
            if current is None:
                previous = None
                continue

            if previous is None:
                samples.append(current)
            else:
                distance = float(np.hypot(current[0] - previous[0], current[1] - previous[1]))
                steps = max(1, int(np.ceil(distance / max(spacing_px, 1e-6))))
                for step_idx in range(1, steps + 1):
                    alpha = step_idx / steps
                    samples.append(
                        (
                            previous[0] + (current[0] - previous[0]) * alpha,
                            previous[1] + (current[1] - previous[1]) * alpha,
                        )
                    )
            previous = current

    if not samples:
        return np.zeros((0, 2), dtype=np.float32)
    return np.asarray(samples, dtype=np.float32)


def score_alignment(
    transform: np.ndarray,
    positions: Dict[str, Sequence[Optional[Tuple[float, float]]]],
    walkable_mask: np.ndarray,
    distance_map: np.ndarray,
    reference_transform: Optional[np.ndarray] = None,
) -> float:
    samples = sample_projected_path_points(transform, positions)
    if samples.size == 0:
        return -1e9

    height, width = walkable_mask.shape
    xs = samples[:, 0]
    ys = samples[:, 1]
    in_bounds = (xs >= 0.0) & (xs <= width - 1) & (ys >= 0.0) & (ys <= height - 1)
    if not np.any(in_bounds):
        return -1e9

    score = -250.0 * float(np.mean(~in_bounds))
    valid = samples[in_bounds]
    x0 = np.floor(valid[:, 0]).astype(np.int32)
    y0 = np.floor(valid[:, 1]).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, width - 1)
    y1 = np.clip(y0 + 1, 0, height - 1)
    dx = valid[:, 0] - x0
    dy = valid[:, 1] - y0

    d00 = distance_map[y0, x0]
    d10 = distance_map[y0, x1]
    d01 = distance_map[y1, x0]
    d11 = distance_map[y1, x1]
    distances = (
        d00 * (1.0 - dx) * (1.0 - dy)
        + d10 * dx * (1.0 - dy)
        + d01 * (1.0 - dx) * dy
        + d11 * dx * dy
    )
    walkable_hits = walkable_mask[np.round(valid[:, 1]).astype(np.int32), np.round(valid[:, 0]).astype(np.int32)]
    score += float(distances.mean())
    score -= 120.0 * float(np.mean(~walkable_hits))

    if reference_transform is not None:
        ref_sx = max(abs(float(reference_transform[0, 0])), 1e-6)
        ref_sy = max(abs(float(reference_transform[1, 1])), 1e-6)
        sx = max(abs(float(transform[0, 0])), 1e-6)
        sy = max(abs(float(transform[1, 1])), 1e-6)
        score -= 35.0 * (np.log(sx / ref_sx) ** 2 + np.log(sy / ref_sy) ** 2)
        score -= 8.0 * (
            ((float(transform[0, 2]) - float(reference_transform[0, 2])) / max(width, 1)) ** 2
            + ((float(transform[1, 2]) - float(reference_transform[1, 2])) / max(height, 1)) ** 2
        )
    return score


def refine_alignment(
    initial_transform: np.ndarray,
    positions: Dict[str, Sequence[Optional[Tuple[float, float]]]],
    walkable_mask: np.ndarray,
    distance_map: np.ndarray,
    max_rounds: int = 8,
) -> Tuple[np.ndarray, float]:
    best = initial_transform.astype(np.float32).copy()
    best_score = score_alignment(best, positions, walkable_mask, distance_map, reference_transform=initial_transform)
    scale_steps = np.array(
        [
            max(abs(best[0, 0]) * 0.15, 0.01),
            max(abs(best[1, 1]) * 0.15, 0.01),
        ],
        dtype=np.float32,
    )
    translate_steps = np.array(
        [
            max(walkable_mask.shape[1] * 0.12, 4.0),
            max(walkable_mask.shape[0] * 0.12, 4.0),
        ],
        dtype=np.float32,
    )

    for _ in range(max_rounds):
        improved = False
        for row, col, step in (
            (0, 0, scale_steps[0]),
            (1, 1, scale_steps[1]),
            (0, 2, translate_steps[0]),
            (1, 2, translate_steps[1]),
        ):
            for delta in (-step, step):
                candidate = best.copy()
                candidate[row, col] += delta
                if (row, col) == (0, 0) and abs(candidate[0, 0]) < 1e-6:
                    continue
                if (row, col) == (1, 1) and abs(candidate[1, 1]) < 1e-6:
                    continue
                candidate_score = score_alignment(
                    candidate,
                    positions,
                    walkable_mask,
                    distance_map,
                    reference_transform=initial_transform,
                )
                if candidate_score > best_score:
                    best = candidate
                    best_score = candidate_score
                    improved = True
        if not improved:
            scale_steps *= 0.5
            translate_steps *= 0.5
            if float(max(scale_steps.max(), translate_steps.max())) < 0.5:
                break
    return best, best_score


def calibrate_trace_to_automap(
    background_rgb: np.ndarray,
    trace: EpisodeTrace,
    marker_points: Optional[Sequence[Tuple[float, float]]] = None,
    env_name: Optional[str] = None,
) -> AutomapCalibration:
    if env_name == "ammo_carrier":
        return calibrate_ammo_carrier_to_automap(background_rgb)

    walkable_mask = build_walkable_mask(background_rgb)
    distance_map = walkable_distance_map(walkable_mask)

    candidates = bbox_alignment_candidates(trace.positions, walkable_mask)
    candidates.extend(spawn_alignment_candidates(trace.positions, marker_points))

    best_transform = None
    best_score = -1e9
    for candidate in candidates:
        refined, refined_score = refine_alignment(candidate, trace.positions, walkable_mask, distance_map)
        if refined_score > best_score:
            best_transform = refined
            best_score = refined_score

    if best_transform is None:
        raise RuntimeError("Could not calibrate telemetry trace against the static automap")

    return AutomapCalibration(
        background_rgb=background_rgb,
        world_to_map=best_transform,
        walkable_mask=walkable_mask,
    )


def resize_automap_calibration(calibration: AutomapCalibration, resolution: str) -> AutomapCalibration:
    out_w, out_h = parse_resolution(resolution)
    bg_h, bg_w = calibration.background_rgb.shape[:2]
    if out_w == bg_w and out_h == bg_h:
        return calibration

    scale_x = out_w / bg_w
    scale_y = out_h / bg_h
    scaled_transform = calibration.world_to_map.copy()
    scaled_transform[0] *= scale_x
    scaled_transform[1] *= scale_y
    resized_bg = cv2.resize(calibration.background_rgb, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    resized_walkable = None
    if calibration.walkable_mask is not None:
        resized_walkable = cv2.resize(
            calibration.walkable_mask.astype(np.uint8),
            (out_w, out_h),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
    return AutomapCalibration(background_rgb=resized_bg, world_to_map=scaled_transform, walkable_mask=resized_walkable)


def project_trace_positions(
    trace: EpisodeTrace,
    cfg: Config,
    calibration: Optional[AutomapCalibration],
) -> Tuple[np.ndarray, Dict[str, List[Optional[Tuple[int, int]]]]]:
    out_w, out_h = parse_resolution(cfg.resolution)
    if calibration is not None:
        calibration = resize_automap_calibration(calibration, cfg.resolution)
        base_map = calibration.background_rgb.copy()
        projected = {
            agent_name: [project_with_transform(calibration.world_to_map, point) for point in path]
            for agent_name, path in trace.positions.items()
        }
        return base_map, projected

    base_map = make_textured_background(out_h, out_w)
    world_bounds = compute_world_bounds(trace.positions)
    projected = {
        agent_name: [world_to_canvas(point, world_bounds, out_w, out_h) for point in path]
        for agent_name, path in trace.positions.items()
    }
    return base_map, projected


def stamp_heatmap(heatmap: np.ndarray, point: Optional[Tuple[int, int]], radius: int) -> None:
    if point is None:
        return
    h, w = heatmap.shape[:2]
    x, y = point
    if 0 <= x < w and 0 <= y < h:
        cv2.circle(heatmap, (x, y), radius, 1.0, thickness=-1)


def _overlay_heatmaps(base: np.ndarray, heatmaps: Dict[str, np.ndarray], alpha_scale: float) -> np.ndarray:
    frame = base.astype(np.float32)
    overlay = np.zeros_like(frame)
    alpha = np.zeros(frame.shape[:2], dtype=np.float32)

    for agent_idx, heatmap in enumerate(heatmaps.values()):
        peak = float(heatmap.max())
        if peak <= 0:
            continue
        intensity = np.log1p(heatmap) / np.log1p(peak)
        overlay += intensity[..., None] * HEATMAP_COLORS[agent_idx % len(HEATMAP_COLORS)]
        alpha = np.maximum(alpha, intensity.astype(np.float32) * alpha_scale)

    blended = frame * (1.0 - alpha[..., None]) + overlay * alpha[..., None]
    return np.clip(blended, 0, 255).astype(np.uint8)


def _valid_polyline(points: Sequence[Optional[Tuple[int, int]]]) -> List[np.ndarray]:
    segments = []
    current: List[Tuple[int, int]] = []
    for point in points:
        if point is None:
            if len(current) >= 2:
                segments.append(np.asarray(current, dtype=np.int32))
            current = []
            continue
        current.append(point)
    if len(current) >= 2:
        segments.append(np.asarray(current, dtype=np.int32))
    return segments


def make_textured_background(height: int, width: int) -> np.ndarray:
    yy, xx = np.indices((height, width))
    background = np.tile(AUTOMAP_BACKGROUND_RGB.reshape(1, 1, 3), (height, width, 1)).astype(np.int16)
    noise = ((xx * 13 + yy * 7) % 9) - 4
    background += noise[..., None]
    minor_grid = ((xx % 72) == 0) | ((yy % 72) == 0)
    major_grid = ((xx % 288) == 0) | ((yy % 288) == 0)
    background[minor_grid] -= np.array([6, 6, 7], dtype=np.int16)
    background[major_grid] -= np.array([12, 12, 14], dtype=np.int16)
    return np.clip(background, 0, 255).astype(np.uint8)


def sample_projected_segment_points(
    start: Optional[Tuple[int, int]],
    end: Optional[Tuple[int, int]],
    spacing_px: float,
) -> np.ndarray:
    if end is None:
        return np.zeros((0, 2), dtype=np.float32)
    if start is None:
        return np.asarray([[float(end[0]), float(end[1])]], dtype=np.float32)

    dx = float(end[0] - start[0])
    dy = float(end[1] - start[1])
    distance = float(np.hypot(dx, dy))
    steps = max(1, int(np.ceil(distance / max(spacing_px, 1e-6))))
    samples = np.zeros((steps, 2), dtype=np.float32)
    for step_idx in range(1, steps + 1):
        alpha = step_idx / steps
        samples[step_idx - 1, 0] = float(start[0] + dx * alpha)
        samples[step_idx - 1, 1] = float(start[1] + dy * alpha)
    return samples


def density_grid_shape(cfg: Config) -> Tuple[int, int]:
    grid_w, grid_h = parse_resolution(cfg.density_grid_resolution)
    assert_matching_automap_aspect(cfg.density_grid_resolution)
    return grid_w, grid_h


def accumulate_density_grid(
    density_grid: np.ndarray,
    segment_points: np.ndarray,
    frame_shape: Tuple[int, int],
) -> None:
    if segment_points.size == 0:
        return

    frame_h, frame_w = frame_shape
    grid_h, grid_w = density_grid.shape
    xs = np.clip((segment_points[:, 0] * grid_w / max(frame_w, 1)).astype(np.int32), 0, grid_w - 1)
    ys = np.clip((segment_points[:, 1] * grid_h / max(frame_h, 1)).astype(np.int32), 0, grid_h - 1)
    np.add.at(density_grid, (ys, xs), 1.0)


def render_density_overlay(
    base_map: np.ndarray,
    density_grid: np.ndarray,
    alpha_scale: float,
    walkable_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    if not np.any(density_grid > 0):
        return base_map.copy()

    out_h, out_w = base_map.shape[:2]
    density = np.log1p(density_grid)
    peak = float(density.max())
    if peak <= 0.0:
        return base_map.copy()

    intensity_small = np.power(density / peak, 0.6).astype(np.float32)
    heat_small = np.clip(np.round(intensity_small * 255.0), 0, 255).astype(np.uint8)
    heat_large = cv2.resize(heat_small, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    colored = cv2.applyColorMap(heat_large, cv2.COLORMAP_INFERNO)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB).astype(np.float32)

    alpha = cv2.resize(intensity_small, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    alpha = np.where(alpha > 0.0, 0.18 + 0.82 * alpha, 0.0) * float(alpha_scale)
    if walkable_mask is not None:
        alpha *= walkable_mask.astype(np.float32)

    blended = base_map.astype(np.float32) * (1.0 - alpha[..., None]) + colored * alpha[..., None]
    return np.clip(blended, 0, 255).astype(np.uint8)


def render_density_heatmap_trace(
    trace: EpisodeTrace,
    cfg: Config,
    calibration: Optional[AutomapCalibration],
) -> Tuple[List[np.ndarray], np.ndarray]:
    base_map, projected = project_trace_positions(trace, cfg, calibration)
    grid_w, grid_h = density_grid_shape(cfg)
    density_grid = np.zeros((grid_h, grid_w), dtype=np.float32)

    walkable_mask = None
    if calibration is not None:
        calibration = resize_automap_calibration(calibration, cfg.resolution)
        walkable_mask = calibration.walkable_mask

    frames: List[np.ndarray] = []
    frame_shape = base_map.shape[:2]
    spacing_px = max(min(frame_shape) / max(grid_w, grid_h), 1.0)

    for frame_idx in range(trace.env_steps):
        for agent_name, path in projected.items():
            current = path[frame_idx]
            previous = path[frame_idx - 1] if frame_idx > 0 else None
            segment_points = sample_projected_segment_points(previous, current, spacing_px=spacing_px)
            accumulate_density_grid(density_grid, segment_points, frame_shape)

        frames.append(render_density_overlay(base_map, density_grid, cfg.density_heatmap_alpha, walkable_mask=walkable_mask))

    return frames, frames[-1].copy()


def render_heatmap_trace(
    trace: EpisodeTrace,
    cfg: Config,
    calibration: Optional[AutomapCalibration],
) -> Tuple[List[np.ndarray], np.ndarray]:
    marker_radius = int(getattr(cfg, "marker_radius", 8))
    base_map, projected = project_trace_positions(trace, cfg, calibration)

    heatmaps = {
        agent_name: np.zeros(base_map.shape[:2], dtype=np.float32)
        for agent_name in trace.positions
    }
    frames: List[np.ndarray] = []

    for frame_idx in range(trace.env_steps):
        for agent_name in trace.positions:
            stamp_heatmap(heatmaps[agent_name], projected[agent_name][frame_idx], cfg.heatmap_radius)

        frame = _overlay_heatmaps(base_map.copy(), heatmaps, cfg.heatmap_alpha)
        for agent_idx, agent_name in enumerate(trace.positions):
            color = tuple(int(channel) for channel in HEATMAP_COLORS[agent_idx % len(HEATMAP_COLORS)])
            for segment in _valid_polyline(projected[agent_name][: frame_idx + 1]):
                cv2.polylines(frame, [segment], False, color, thickness=cfg.line_thickness, lineType=cv2.LINE_AA)

            point = projected[agent_name][frame_idx]
            if point is not None:
                cv2.circle(frame, point, marker_radius, color, thickness=-1, lineType=cv2.LINE_AA)
                cv2.circle(frame, point, marker_radius + 3, (255, 255, 255), thickness=2, lineType=cv2.LINE_AA)

        frames.append(frame)

    return frames, frames[-1].copy()


def run_episode(
    cfg: Config,
    env,
    env_info,
    actor_critic,
    device: torch.device,
    obs,
    rnn_states: torch.Tensor,
) -> Tuple[dict, torch.Tensor, EpisodeTrace]:
    num_agents = env.num_agents
    agent_names = [f"agent{i}" for i in range(num_agents)]
    start_infos = env.unwrapped.info()
    last_positions = extract_positions_from_infos(start_infos)
    traces = {agent_name: [last_positions.get(agent_name)] for agent_name in agent_names}

    action_mask = obs.pop("action_mask").to(device) if "action_mask" in obs else None
    done_flags = [False for _ in range(num_agents)]
    episode_reward = torch.zeros(num_agents, dtype=torch.float32)
    env_steps = 1

    while not all(done_flags):
        normalized_obs = prepare_and_normalize_obs(actor_critic, obs)
        policy_outputs = actor_critic(normalized_obs, rnn_states, action_mask=action_mask)
        actions = policy_outputs["actions"]

        if cfg.eval_deterministic:
            action_distribution = actor_critic.action_distribution()
            actions = reshape_deterministic_actions(actions, argmax_actions(action_distribution))

        if actions.ndim == 1:
            actions = unsqueeze_tensor(actions, dim=-1)
        actions = preprocess_actions(env_info, actions)
        rnn_states = policy_outputs["new_rnn_states"]

        for _ in range(cfg.render_action_repeat):
            obs, rew, terminated, truncated, infos = env.step(actions)
            infos = infos or [{} for _ in range(num_agents)]
            action_mask = obs.pop("action_mask").to(device) if "action_mask" in obs else None

            dones = make_dones(terminated, truncated).cpu().numpy()
            episode_reward += rew.float()
            positions = extract_positions_from_infos(infos, last_positions)
            for agent_name, point in positions.items():
                traces[agent_name].append(point)
                if point is not None:
                    last_positions[agent_name] = point

            env_steps += 1
            for agent_i, done in enumerate(dones):
                if done:
                    done_flags[agent_i] = True
                    rnn_states[agent_i] = torch.zeros([get_rnn_size(cfg)], dtype=torch.float32, device=device)

            if all(dones):
                break

    trace = EpisodeTrace(
        positions=traces,
        rewards=episode_reward.cpu().numpy(),
        env_steps=env_steps,
    )
    return obs, rnn_states, trace


def record_topdown_heatmaps(cfg: Config) -> Tuple[StatusCode, float]:
    cfg = load_from_checkpoint(cfg)
    cfg.load_checkpoint_kind = "best"
    register_model_factory(cfg)
    assert_matching_automap_aspect(cfg.resolution)

    cfg.eval_env_frameskip = 1
    eval_env_frameskip = cfg.eval_env_frameskip
    assert cfg.env_frameskip % eval_env_frameskip == 0, (
        f"{cfg.env_frameskip=} must be divisible by {eval_env_frameskip=}"
    )
    cfg.render_action_repeat = cfg.env_frameskip // eval_env_frameskip
    cfg.env_frameskip = cfg.eval_env_frameskip = eval_env_frameskip
    cfg.num_envs = 1
    cfg.with_wandb = False
    cfg.wandb_record_every = 0
    cfg.video_fps = int(cfg.video_fps)

    os.makedirs(cfg.output_dir, exist_ok=True)

    env = make_env(cfg)
    env_info = extract_env_info(env, cfg)
    if hasattr(env.unwrapped, "reset_on_init"):
        env.unwrapped.reset_on_init = False

    device = torch.device("cpu" if cfg.device == "cpu" else "cuda")
    actor_critic = load_policy(cfg, env, device)

    obs, _ = env.reset()
    static_background_rgb, marker_points = capture_static_automap(cfg.env)

    rewards = []
    rnn_states = torch.zeros([env.num_agents, get_rnn_size(cfg)], dtype=torch.float32, device=device)
    with torch.no_grad():
        for episode_idx in range(1, cfg.max_num_episodes + 1):
            stem = output_stem(cfg, episode_idx)
            mp4_path = os.path.join(cfg.output_dir, f"{stem}.mp4")
            png_path = os.path.join(cfg.output_dir, f"{stem}.png")
            density_stem = density_output_stem(cfg, episode_idx)
            density_mp4_path = os.path.join(cfg.output_dir, f"{density_stem}.mp4")
            density_png_path = os.path.join(cfg.output_dir, f"{density_stem}.png")

            if not cfg.overwrite and (
                os.path.exists(mp4_path)
                or os.path.exists(png_path)
                or os.path.exists(density_mp4_path)
                or os.path.exists(density_png_path)
            ):
                raise FileExistsError(f"Output already exists for {stem}; rerun with --overwrite")

            obs, rnn_states, trace = run_episode(cfg, env, env_info, actor_critic, device, obs, rnn_states)
            calibration = calibrate_trace_to_automap(
                static_background_rgb,
                trace,
                marker_points=marker_points,
                env_name=cfg.env,
            )
            frames, final_frame = render_heatmap_trace(trace, cfg, calibration)
            density_frames, density_final_frame = render_density_heatmap_trace(trace, cfg, calibration)
            save_video(frames, mp4_path, cfg.video_fps)
            save_image(final_frame, png_path)
            save_video(density_frames, density_mp4_path, cfg.video_fps)
            save_image(density_final_frame, density_png_path)
            rewards.append(float(trace.rewards.mean()))

    env.close()
    avg_reward = float(np.mean(rewards)) if rewards else 0.0
    log.info(
        "Finished %d episodes for %s (%s). Avg per-agent reward %.3f",
        cfg.max_num_episodes,
        cfg.env,
        algo_label(cfg),
        avg_reward,
    )
    return ExperimentStatus.SUCCESS, avg_reward


def main():
    register_vizdoom_components()
    cfg = parse_topdown_args()
    status, _ = record_topdown_heatmaps(cfg)
    return status


if __name__ == "__main__":
    sys.exit(main())
