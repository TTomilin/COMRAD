from __future__ import annotations

from typing import NamedTuple, Optional

HIDDEN_RESET_COUNT_KEY = "_hidden_reset_count"


def hidden_reset_count(infos) -> int:
    """Episodes MultiAgentEnv crash recovery restarted underneath a step (`_hidden_reset_count`, 0 when none)."""
    if isinstance(infos, dict):
        return int(infos.get(HIDDEN_RESET_COUNT_KEY, 0))
    if isinstance(infos, (list, tuple)):
        return max((int(info.get(HIDDEN_RESET_COUNT_KEY, 0)) for info in infos if isinstance(info, dict)), default=0)
    return 0


class NewEpisode(NamedTuple):
    """The episode the observation of a joint step belongs to, when it is not the episode the step started in."""

    steps: int  # joint env steps the observation is into the new episode
    infos: list  # per-agent infos describing the observation


def joint_new_episode(terminated, truncated, infos) -> Optional[NewEpisode]:
    """Whether a joint (MultiAgentEnv) step returned the observation of a new episode, and how to read it.

    Auto-reset: every agent is done and every info carries `reset_info`; the observation is the new episode's first
    frame, described by the `reset_info`s.
    Crash recovery: the game group was restarted underneath the step (`_hidden_reset_count`); the done flags stay
    False and the observation is one step into the new episode, described by `infos` themselves.
    """
    if not isinstance(infos, (list, tuple)):
        return None
    if all(bool(t) or bool(tr) for t, tr in zip(terminated, truncated)) and all(
        isinstance(info, dict) and isinstance(info.get("reset_info"), dict) for info in infos
    ):
        return NewEpisode(steps=0, infos=[info["reset_info"] for info in infos])
    if hidden_reset_count(infos) > 0:
        return NewEpisode(steps=1, infos=list(infos))
    return None
