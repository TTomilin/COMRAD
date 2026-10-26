# `comrad/curriculum/`

Curricula over pools of pre-generated WADs, enabled with `--wad_batch=<dir> --curriculum=<strategy>` ([docs](https://ttomilin.github.io/COMRAD/training/reproducing/#wad-batches-and-curricula)).

- Strategies: `uniform.py`, `sequential.py`, `learning_progress.py`, `omni.py`, `plr.py`
- `base.py`: shared task-pool interface
- `observer.py`: logging and persistence. Curriculum state is saved in learner checkpoints and in `curriculum_state.json` in the experiment directory.
