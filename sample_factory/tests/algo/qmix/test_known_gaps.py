from __future__ import annotations

import pytest
import gymnasium as gym

from sf.doom.doom_utils import DOOM_ENVS


class TestRewardShapingMissesTeamTermination:

    @pytest.mark.xfail(
        reason="Per-agent reward shaping runs before MultiAgentEnv team termination",
        strict=True,
    )
    def test_surviving_agent_should_get_true_objective_on_team_death(self):
        """When agent A dies and wipe_when_one_die terminates agent B,
        agent B's wrapper should still set true_objective."""
        from sf.doom.wrappers.scenario_wrappers.pitfall_reward_shaping import DoomPitfallRewardShaping

        class FakeEnv:
            def __init__(self):
                self.observation_space = None
                self.action_space = None
                self._step_result = None

            def step(self, action):
                return self._step_result

            def reset(self, **kwargs):
                return {}, {"POSITION_X": 32.0, "DEAD": 0}

        # Agent B's wrapper (agent B is alive, agent A already died)
        fake_env = FakeEnv()
        wrapper = DoomPitfallRewardShaping(
            fake_env, goal_x=100.0, goal_reward=10.0,
        )
        wrapper.reset()

        fake_env._step_result = (
            {}, # obs
            0.0, # reward
            False, # terminated — wrapper sees False (agent B is alive)
            False, # truncated
            {"POSITION_X": 100.0, "DEAD": 0},  # At goal_x
        )
        _, _, _, _, infos = wrapper.step(0)
        # Goal flag is now set in the wrapper

        # Now agent A dies → wipe_when_one_die will terminate agent B.
        # But per-agent step runs BEFORE wipe_when_one_die.
        # Agent B's wrapper sees done=False (not yet team-terminated).
        fake_env._step_result = (
            {}, # obs
            0.0, # reward
            False, # terminated — wrapper sees False (team termination hasn't happened yet)
            False, # truncated
            {"POSITION_X": 100.0, "DEAD": 0},
        )
        _, _, _, _, infos = wrapper.step(0)

        # After MultiAgentEnv.wipe_when_one_die runs, agent B IS terminated.
        # But the wrapper already ran with done=False, so true_objective is NOT set
        # on this step. The wrapper would only set it on the NEXT reset or if
        # done=True were passed.
        assert "true_objective" in infos, (
            "Surviving agent should have true_objective set when team is terminated, "
            "but per-agent wrapper ran before team termination was applied"
        )


class TestSingleHeadBranchInactive:

    @pytest.mark.xfail(
        reason="All current Doom envs use multi-head Tuple action spaces",
        strict=True,
    )
    def test_any_doom_env_uses_single_head(self):
        single_head_envs = []
        for spec in DOOM_ENVS:
            action_space = spec.action_space
            if isinstance(action_space, gym.spaces.Discrete):
                single_head_envs.append(spec.name)
            elif isinstance(action_space, gym.spaces.Tuple) and len(action_space.spaces) == 1:
                single_head_envs.append(spec.name)

        assert len(single_head_envs) > 0, (
            f"No Doom env uses single-head action space. "
            f"All {len(DOOM_ENVS)} envs use multi-head Tuple spaces, "
            f"making the single-head branch in get_q_for_actions dead code."
        )
