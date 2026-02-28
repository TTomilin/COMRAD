import sys

from sample_factory.enjoy import enjoy
from comrad.train import parse_args, register_vizdoom_components, register_model_factory


def main():
    register_vizdoom_components()
    cfg = parse_args(evaluation=True)

    # num_agents may be -1 (default value)
    if cfg.num_agents < 1:
        from comrad.utils.doom_utils import get_num_agents
        cfg.num_agents = get_num_agents(cfg, cfg.env)

    register_model_factory(cfg)
    status = enjoy(cfg)
    return status


if __name__ == "__main__":
    sys.exit(main())
