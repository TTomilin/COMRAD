from types import SimpleNamespace

import comrad.train as comrad_train
import sample_factory.train as sf_train
from sample_factory.algo.utils.misc import ExperimentStatus
from sample_factory.utils.attr_dict import AttrDict


def test_prepare_cfg_for_training_loads_checkpoint_before_batch_registration(monkeypatch):
    cli_cfg = SimpleNamespace(
        restart_behavior="resume",
        wad_batch=None,
        env="doom_pitfall",
        num_agents=-1,
        experiment="resume_test",
        train_dir="/tmp",
    )
    loaded_cfg = SimpleNamespace(
        restart_behavior="resume",
        wad_batch="saved_batch_dir",
        env="doom_pitfall",
        num_agents=-1,
        experiment="resume_test",
        train_dir="/tmp",
    )
    curriculum = object()
    calls = []

    def fake_load_from_checkpoint(cfg):
        calls.append(("load", cfg))
        return loaded_cfg

    def fake_register_batch_env(cfg):
        calls.append(("register", cfg.wad_batch, cfg.env))
        return "doom_pitfall_batch", curriculum

    monkeypatch.setattr(comrad_train, "maybe_load_from_checkpoint", fake_load_from_checkpoint)
    monkeypatch.setattr(comrad_train, "register_batch_env", fake_register_batch_env)
    monkeypatch.setattr(comrad_train, "get_num_agents", lambda cfg, env_name: 3)

    cfg = comrad_train.prepare_cfg_for_training(cli_cfg)

    assert cfg is loaded_cfg
    assert cfg.env == "doom_pitfall_batch"
    assert cfg._curriculum is curriculum
    assert cfg.num_agents == 3
    assert calls == [
        ("load", cli_cfg),
        ("register", "saved_batch_dir", "doom_pitfall"),
    ]


def test_register_batch_env_does_not_append_batch_suffix_twice(monkeypatch):
    captured = {}

    def fake_doom_env_by_name(env_name):
        captured["base_env"] = env_name
        return SimpleNamespace(name=env_name)

    monkeypatch.setattr(comrad_train, "doom_env_by_name", fake_doom_env_by_name)
    monkeypatch.setattr(
        comrad_train,
        "register_env",
        lambda env_name, make_env_func: captured.update({"env_name": env_name}),
    )

    import comrad.curriculum as comrad_curriculum
    import comrad.envs.wad_catalog as wad_catalog

    monkeypatch.setattr(comrad_curriculum, "make_curriculum", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        wad_catalog.WadBatch,
        "from_dir",
        staticmethod(lambda _: SimpleNamespace(entries=[SimpleNamespace(name="wad_a")])),
    )

    cfg = SimpleNamespace(
        env="doom_pitfall_batch",
        wad_batch="saved_batch_dir",
        curriculum="uniform",
        algo="APPO",
        seed=7,
    )

    env_name, _ = comrad_train.register_batch_env(cfg)

    assert env_name == "doom_pitfall_batch"
    assert captured["env_name"] == "doom_pitfall_batch"
    assert captured["base_env"] == "doom_pitfall"


def test_make_runner_can_skip_checkpoint_reload(monkeypatch):
    cfg = SimpleNamespace(restart_behavior="resume", serial_mode=True, with_pbt=False)
    load_calls = []

    def fake_load_from_checkpoint(_cfg):
        load_calls.append(True)
        raise AssertionError("make_runner should not reload checkpoint config")

    class DummyRunner:
        def __init__(self, runner_cfg):
            self.cfg = runner_cfg

    monkeypatch.setattr(sf_train, "maybe_load_from_checkpoint", fake_load_from_checkpoint)
    monkeypatch.setattr(sf_train, "SerialRunner", DummyRunner)

    runner_cfg, runner = sf_train.make_runner(cfg, load_checkpoint_cfg=False)

    assert runner_cfg is cfg
    assert runner.cfg is cfg
    assert load_calls == []


def test_main_removes_curriculum_from_attrdict_cfg(monkeypatch):
    curriculum = object()
    cfg = AttrDict(
        restart_behavior="restart",
        serial_mode=True,
        with_pbt=False,
        num_agents=1,
        with_wandb=False,
        wandb_record_every=0,
        _curriculum=curriculum,
    )

    class DummyRunner:
        def __init__(self, runner_cfg):
            self.cfg = runner_cfg
            self.learners = {}
            self.observers = []
            self.run_called = False

        def register_observer(self, observer):
            self.observers.append(observer)

        def init(self):
            return ExperimentStatus.SUCCESS

        def run(self):
            self.run_called = True
            return ExperimentStatus.SUCCESS

    runner = DummyRunner(cfg)

    monkeypatch.setattr(comrad_train, "register_vizdoom_components", lambda: None)
    monkeypatch.setattr(comrad_train, "parse_args", lambda: cfg)
    monkeypatch.setattr(comrad_train, "prepare_cfg_for_training", lambda parsed_cfg: parsed_cfg)
    monkeypatch.setattr(comrad_train, "make_runner", lambda runner_cfg, load_checkpoint_cfg=False: (runner_cfg, runner))

    status = comrad_train.main()

    assert status == ExperimentStatus.SUCCESS
    assert "_curriculum" not in cfg
    assert runner.run_called is True
    assert len(runner.observers) == 1
