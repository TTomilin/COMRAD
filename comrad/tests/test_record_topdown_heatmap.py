from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch

from comrad.record_topdown_heatmap import (
    AUTOMAP_BACKGROUND_RGB,
    AUTOMAP_PLAYER0_RGB,
    AUTOMAP_PLAYER1_RGB,
    AutomapCalibration,
    AMMO_CARRIER_DEPOT_CENTER_WORLD_POINT,
    EpisodeTrace,
    assert_matching_automap_aspect,
    build_walkable_mask,
    calibrate_trace_to_automap,
    density_grid_shape,
    detect_start_markers,
    fit_spawn_alignment,
    project_with_transform,
    render_density_heatmap_trace,
    render_heatmap_trace,
    reshape_deterministic_actions,
    sample_projected_path_points,
    scrub_colored_automap_artifacts,
    scrub_detected_markers,
)


def _cfg(resolution: str = "1600x1200") -> SimpleNamespace:
    return SimpleNamespace(
        resolution=resolution,
        heatmap_radius=10,
        heatmap_alpha=0.7,
        line_thickness=3,
        marker_radius=6,
        density_grid_resolution="40x30",
        density_heatmap_alpha=0.8,
    )


def test_assert_matching_automap_aspect_rejects_distortion():
    assert_matching_automap_aspect("1600x1200")
    assert_matching_automap_aspect("1024x768")

    with pytest.raises(ValueError, match="aspect ratio"):
        assert_matching_automap_aspect("1280x720")


def test_detect_start_markers_prefers_expected_automap_colors():
    image = np.tile(AUTOMAP_BACKGROUND_RGB.reshape(1, 1, 3), (160, 240, 1))

    cv2.circle(image, (40, 90), 8, tuple(int(x) for x in AUTOMAP_PLAYER0_RGB), thickness=-1)
    cv2.circle(image, (190, 60), 8, tuple(int(x) for x in AUTOMAP_PLAYER1_RGB), thickness=-1)
    cv2.circle(image, (120, 120), 10, (255, 220, 0), thickness=-1)

    detections = detect_start_markers(image)

    assert len(detections) == 2
    assert detections[0].centroid == pytest.approx((40, 90), abs=2.0)
    assert detections[1].centroid == pytest.approx((190, 60), abs=2.0)


def test_scrub_detected_markers_removes_player_markers():
    image = np.tile(AUTOMAP_BACKGROUND_RGB.reshape(1, 1, 3), (120, 180, 1))
    cv2.circle(image, (30, 60), 8, tuple(int(x) for x in AUTOMAP_PLAYER0_RGB), thickness=-1)
    cv2.circle(image, (140, 50), 8, tuple(int(x) for x in AUTOMAP_PLAYER1_RGB), thickness=-1)

    detections = detect_start_markers(image)
    scrubbed = scrub_detected_markers(image, detections)

    assert np.any(np.all(image == AUTOMAP_PLAYER0_RGB, axis=2))
    assert np.any(np.all(image == AUTOMAP_PLAYER1_RGB, axis=2))
    assert not np.any(np.all(scrubbed == AUTOMAP_PLAYER0_RGB, axis=2))
    assert not np.any(np.all(scrubbed == AUTOMAP_PLAYER1_RGB, axis=2))


def test_scrub_colored_automap_artifacts_removes_marker_like_shapes():
    image = np.tile(AUTOMAP_BACKGROUND_RGB.reshape(1, 1, 3), (120, 180, 1))
    cv2.rectangle(image, (20, 50), (40, 70), (80, 80, 80), thickness=2)
    cv2.drawMarker(image, (90, 60), (255, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=14, thickness=2)
    cv2.drawMarker(image, (130, 55), (95, 207, 87), markerType=cv2.MARKER_TRIANGLE_UP, markerSize=14, thickness=2)

    scrubbed = scrub_colored_automap_artifacts(image)

    assert np.any(np.all(image == np.array([255, 0, 255], dtype=np.uint8), axis=2))
    assert np.any(np.all(image == np.array([95, 207, 87], dtype=np.uint8), axis=2))
    assert not np.any(np.all(scrubbed == np.array([255, 0, 255], dtype=np.uint8), axis=2))
    assert not np.any(np.all(scrubbed == np.array([95, 207, 87], dtype=np.uint8), axis=2))
    assert np.any(np.all(scrubbed == np.array([80, 80, 80], dtype=np.uint8), axis=2))


def test_build_walkable_mask_excludes_outside_background():
    image = np.tile(AUTOMAP_BACKGROUND_RGB.reshape(1, 1, 3), (120, 180, 1))
    wall_color = (40, 40, 40)
    cv2.rectangle(image, (20, 20), (160, 100), wall_color, thickness=2)

    walkable = build_walkable_mask(image, wall_threshold=10, wall_padding=1)

    assert walkable[60, 90]
    assert not walkable[5, 5]


def test_fit_spawn_alignment_preserves_expected_projection():
    transform = fit_spawn_alignment(
        world_points=[(0.0, 0.0), (10.0, 20.0)],
        map_points=[(100.0, 300.0), (300.0, 100.0)],
    )

    assert project_with_transform(transform, (0.0, 0.0)) == (100, 300)
    assert project_with_transform(transform, (10.0, 20.0)) == (300, 100)
    assert project_with_transform(transform, (5.0, 10.0)) == (200, 200)


def test_calibrate_trace_to_automap_uses_geometry_not_marker_guess():
    image = np.tile(AUTOMAP_BACKGROUND_RGB.reshape(1, 1, 3), (180, 260, 1))
    wall_color = (40, 40, 40)
    cv2.rectangle(image, (20, 20), (120, 160), wall_color, thickness=2)
    cv2.rectangle(image, (140, 40), (240, 140), wall_color, thickness=2)
    cv2.rectangle(image, (118, 85), (142, 95), wall_color, thickness=2)

    expected_pixels = [
        (40.0, 120.0),
        (80.0, 120.0),
        (100.0, 95.0),
        (130.0, 90.0),
        (160.0, 70.0),
        (210.0, 70.0),
    ]
    world_path = [((px - 30.0) / 2.0, (200.0 - py) / 3.0) for px, py in expected_pixels]
    trace = EpisodeTrace(
        positions={"agent0": world_path, "agent1": list(reversed(world_path))},
        rewards=np.zeros(2, dtype=np.float32),
        env_steps=len(world_path),
    )

    calibration = calibrate_trace_to_automap(
        image,
        trace,
        marker_points=[expected_pixels[0], expected_pixels[-1]],
    )
    samples = sample_projected_path_points(calibration.world_to_map, trace.positions, spacing_px=2.0)
    assert samples.shape[0] > len(world_path)
    rounded_x = np.clip(np.round(samples[:, 0]).astype(np.int32), 0, calibration.walkable_mask.shape[1] - 1)
    rounded_y = np.clip(np.round(samples[:, 1]).astype(np.int32), 0, calibration.walkable_mask.shape[0] - 1)
    inside_walkable = calibration.walkable_mask[rounded_y, rounded_x]
    assert inside_walkable.mean() > 0.85


def test_reshape_deterministic_actions_restores_sampled_layout():
    sampled = torch.zeros((2, 4), dtype=torch.int64)
    deterministic = torch.arange(8, dtype=torch.int64).reshape(1, 8)

    reshaped = reshape_deterministic_actions(sampled, deterministic)

    assert tuple(reshaped.shape) == (2, 4)
    assert torch.equal(
        reshaped,
        torch.tensor(
            [
                [0, 2, 4, 6],
                [1, 3, 5, 7],
            ],
            dtype=torch.int64,
        ),
    )


def _synthetic_ammo_carrier_automap() -> np.ndarray:
    image = np.tile(AUTOMAP_BACKGROUND_RGB.reshape(1, 1, 3), (240, 420, 1))
    wall_color = (40, 40, 40)

    cv2.rectangle(image, (25, 25), (235, 215), wall_color, thickness=2)
    cv2.rectangle(image, (285, 55), (395, 185), wall_color, thickness=2)
    cv2.rectangle(image, (235, 112), (285, 128), wall_color, thickness=2)

    for center in ((85, 75), (175, 75), (85, 165), (175, 165)):
        cv2.rectangle(
            image,
            (center[0] - 16, center[1] - 16),
            (center[0] + 16, center[1] + 16),
            wall_color,
            thickness=2,
        )

    return image


def test_calibrate_trace_to_automap_uses_ammo_carrier_geometry_not_random_markers():
    image = _synthetic_ammo_carrier_automap()
    trace = EpisodeTrace(
        positions={
            "agent0": [(0.0, 0.0), (200.0, 0.0), (500.0, 0.0)],
            "agent1": [(900.0, 0.0), (960.0, 0.0), (900.0, 100.0)],
        },
        rewards=np.zeros(2, dtype=np.float32),
        env_steps=3,
    )

    calibration = calibrate_trace_to_automap(
        image,
        trace,
        marker_points=[(320.0, 140.0), (360.0, 140.0)],
        env_name="ammo_carrier",
    )

    assert project_with_transform(calibration.world_to_map, (0.0, 0.0)) == pytest.approx((130, 120), abs=6)
    assert project_with_transform(calibration.world_to_map, AMMO_CARRIER_DEPOT_CENTER_WORLD_POINT) == pytest.approx(
        (340, 120), abs=6
    )


def test_render_heatmap_trace_matches_automap_resolution():
    background = np.tile(AUTOMAP_BACKGROUND_RGB.reshape(1, 1, 3), (120, 160, 1))
    calibration = AutomapCalibration(
        background_rgb=background,
        world_to_map=np.array([[1.0, 0.0, 20.0], [0.0, 1.0, 15.0]], dtype=np.float32),
    )
    trace = EpisodeTrace(
        positions={
            "agent0": [(0.0, 0.0), (20.0, 10.0), (40.0, 20.0)],
            "agent1": [(80.0, 10.0), (70.0, 20.0), (60.0, 30.0)],
        },
        rewards=np.array([1.0, 2.0], dtype=np.float32),
        env_steps=3,
    )

    frames, final_frame = render_heatmap_trace(trace, _cfg("160x120"), calibration)

    assert len(frames) == 3
    assert all(frame.shape == (120, 160, 3) for frame in frames)
    assert final_frame.shape == (120, 160, 3)
    assert np.any(final_frame != background)


def test_density_grid_shape_rejects_distorted_resolution():
    assert density_grid_shape(_cfg("160x120")) == (40, 30)

    bad_cfg = _cfg("160x120")
    bad_cfg.density_grid_resolution = "40x40"
    with pytest.raises(ValueError, match="aspect ratio"):
        density_grid_shape(bad_cfg)


def test_render_density_heatmap_trace_accumulates_visits():
    background = np.tile(AUTOMAP_BACKGROUND_RGB.reshape(1, 1, 3), (120, 160, 1))
    calibration = AutomapCalibration(
        background_rgb=background,
        world_to_map=np.array([[1.0, 0.0, 20.0], [0.0, 1.0, 15.0]], dtype=np.float32),
    )
    trace = EpisodeTrace(
        positions={
            "agent0": [(0.0, 0.0), (10.0, 0.0), (20.0, 0.0), (20.0, 0.0)],
            "agent1": [(0.0, 0.0), (10.0, 0.0), (20.0, 0.0), (80.0, 60.0)],
        },
        rewards=np.zeros(2, dtype=np.float32),
        env_steps=4,
    )

    frames, final_frame = render_density_heatmap_trace(trace, _cfg("160x120"), calibration)

    assert len(frames) == 4
    assert all(frame.shape == (120, 160, 3) for frame in frames)
    assert final_frame.shape == (120, 160, 3)
    repeated_visit_patch = final_frame[8:36, 18:48]
    sparse_patch = final_frame[60:100, 90:140]
    assert np.abs(repeated_visit_patch.astype(np.int16) - background[8:36, 18:48].astype(np.int16)).mean() > 5.0
    assert np.abs(repeated_visit_patch.astype(np.int16) - background[8:36, 18:48].astype(np.int16)).mean() > np.abs(
        sparse_patch.astype(np.int16) - background[60:100, 90:140].astype(np.int16)
    ).mean()
