"""Entrypoint: dispatch on config.MODE."""

from . import config as cfg


def main():
    if cfg.MODE == "bench":
        from . import bench

        bench.main()
    elif cfg.MODE == "fvdeval":
        from . import fvdeval

        fvdeval.main()
    else:
        from . import train

        train.main()


if __name__ == "__main__":
    main()
