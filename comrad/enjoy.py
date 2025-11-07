import sys

from sample_factory.enjoy import enjoy
from sf.train import parse_args, register_vizdoom_components


def main():
    register_vizdoom_components()
    cfg = parse_args(evaluation=True)
    status = enjoy(cfg)
    return status


if __name__ == "__main__":
    sys.exit(main())