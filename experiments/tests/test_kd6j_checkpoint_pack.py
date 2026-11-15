import numpy as np
import pytest
import torch

import experiments.iclr2027_rebuttal.eval_frozen_partner as efp
import experiments.iclr2027_rebuttal.eval_kd6j_checkpoint_pack as pack
from experiments.tests.test_frozen_partner_eval import ENTRY, FakeTwoAgentEnv, HiddenResetEnv, _policy, _real_model


class OcclusionEnv(FakeTwoAgentEnv):
    """Renders the partner (pixel value 200) unless `partner_occlusion_mode` hides it (pixel value 50)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.partner_occlusion_mode = "none"
        self.modes_at_reset = []

    def _obs(self):
        value = 200 if self.partner_occlusion_mode == "none" else 50
        return {"obs": torch.full((2, 3, 72, 128), value, dtype=torch.uint8)}

    def reset(self):
        self.modes_at_reset.append(self.partner_occlusion_mode)
        return super().reset()


class FrameRecorder:
    def __init__(self, model):
        self.model = model
        self.frames = []

    def __getattr__(self, name):
        return getattr(self.model, name)

    def __call__(self, obs, rnn_states, **kwargs):
        self.frames.append(obs["obs"].clone())
        return self.model(obs, rnn_states, **kwargs)


def _modes(env, model, env_info, cfg, episodes=2):
    common = (env, model, env_info, cfg, _policy("coop_puzzle", "MAPPO"), ENTRY, episodes, torch.device("cpu"))
    return [
        pack.run_mode(*common, mode="pixel_partial", pass_index=0, partner_occlusion_mode="none", blind_horizon=None),
        pack.run_mode(*common, mode="partner_occlusion", pass_index=1, partner_occlusion_mode="both", blind_horizon=None),
        pack.run_mode(*common, mode="blind_eval", pass_index=2, partner_occlusion_mode="none", blind_horizon=10),
    ]


def test_every_intervention_pass_starts_from_its_own_masked_reset():
    cfg, model, env_info = _real_model("MAPPO", "coop_puzzle")
    env = OcclusionEnv(ep_len=3)
    recorder = FrameRecorder(model)
    normal, occluded, blind = _modes(env, recorder, env_info, cfg)
    assert env.modes_at_reset == ["none", "both", "none"]
    assert [len(r) for r in (normal, occluded, blind)] == [2, 2, 2]
    assert [r["mode"] for r in normal + occluded + blind] == ["pixel_partial"] * 2 + ["partner_occlusion"] * 2 + ["blind_eval"] * 2

    frames = recorder.frames
    first_pixel = [float(f.reshape(2, -1)[0, 1]) for f in frames]
    # 6 steps per pass; step 0 of the occlusion pass already sees the occluded frame, step 0 of the blind pass a black one
    assert len(set(first_pixel[0:6])) == 1 and len(set(first_pixel[6:12])) == 1
    assert first_pixel[0] != first_pixel[6]
    black = frames[12].reshape(2, -1)[:, 1:]
    assert torch.all(black == black.flatten()[0])
    assert all(torch.equal(f.reshape(2, -1)[:, 1:], black) for f in frames[12:])


def test_blind_observation_encodes_episode_progress_like_the_blind_wrapper():
    transform = pack.blind_observation(horizon=10)
    obs = {"obs": torch.full((2, 3, 4, 4), 77, dtype=torch.uint8), "measurements": torch.ones(2, 3)}
    out = transform(obs, 5)
    flat = out["obs"].reshape(2, -1)
    assert flat[:, 0].tolist() == [128, 128] and int(flat[:, 1:].sum()) == 0
    assert torch.equal(out["measurements"], obs["measurements"])
    assert float(transform({"obs": torch.ones(1, 4)}, 30)["obs"][0, 0]) == 1.0


def test_crashed_episodes_are_discarded_not_continued():
    cfg, model, env_info = _real_model("MAPPO", "coop_puzzle")

    class CrashingOcclusionEnv(HiddenResetEnv, OcclusionEnv):
        pass

    with pytest.raises(efp.HiddenResetError):
        pack.run_mode(
            CrashingOcclusionEnv(ep_len=4), model, env_info, cfg, _policy("coop_puzzle", "MAPPO"), ENTRY, 1,
            torch.device("cpu"), mode="pixel_partial", pass_index=0, partner_occlusion_mode="none", blind_horizon=None,
        )


def test_missing_visibility_wrapper_is_an_error():
    cfg, model, env_info = _real_model("MAPPO", "coop_puzzle")
    with pytest.raises(RuntimeError, match="VisibilityDiagnosticsWrapper"):
        pack.run_mode(
            FakeTwoAgentEnv(), model, env_info, cfg, _policy("coop_puzzle", "MAPPO"), ENTRY, 1, torch.device("cpu"),
            mode="pixel_partial", pass_index=0, partner_occlusion_mode="none", blind_horizon=None,
        )


def test_passes_are_reproducible():
    cfg, model, env_info = _real_model("MAPPO", "coop_puzzle")
    runs = []
    for _ in range(2):
        env = OcclusionEnv(ep_len=3)
        _modes(env, model, env_info, cfg)
        runs.append(np.stack(env.actions))
    assert np.array_equal(runs[0], runs[1])
