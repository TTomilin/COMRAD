import os
import numpy as np
import logging
import wandb
from sample_factory.algo.utils.misc import EPISODIC

log = logging.getLogger(__name__)

def upload_video(runner, cfg):
    def upload(_runner, msg, policy_id):
        stats = msg.get(EPISODIC)
        if not isinstance(stats, dict): return
        extra_stats = stats.get("episode_extra_stats")
        if not isinstance(extra_stats, dict): return
        data = extra_stats.pop("wandb_video", None)
        if not isinstance(data, dict): return # for when wandb_mode=offline

        path = data.get("path")
        if not path: return

        try:
            with np.load(path) as p:
                frames = np.asarray(p["frames"])
            os.remove(path)

            ep = data.get("episode", 0)
            fps = data.get("fps", getattr(cfg, "wandb_video_fps", 35))

            # worker_index = data.get("worker_index")
            # vector_index = data.get("vector_index")
            # key = f"videos/p_{policy_id:02d}"
            # if worker_index is not None:
            #     key += f"_w_{int(worker_index):02d}"
            # if vector_index is not None:
            #     key += f"_v_{int(vector_index):02d}"
            # key += f"_ep_{ep:05d}"
            # wandb.log({key: wandb.Video(frames, fps=fps, format="mp4")}, step=None and _runner.env_steps.get(policy_id, 0))

            key = f"videos/p_{policy_id:02d}_ep_{ep:05d}"
            wandb.log({key: wandb.Video(frames, fps=fps, format="mp4")}, step=None)
            # step None so only logs 1 latest video
        except Exception as e:
            # happen when vizdoom crashes while video is being encoded
            log.debug(f"Failed to upload video to wandb (likely during shutdown): {e}")
            if os.path.exists(path):
                try:
                    os.remove(path)
                except Exception:
                    pass

    runner.policy_msg_handlers.setdefault(EPISODIC, []).insert(0, upload)
