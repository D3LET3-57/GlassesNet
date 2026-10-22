"""CLI entry point for a single training run.

Example:
    python -m src.run --config experiments/configs/final.json --fold 0 \
        --run-name cv_final_fold0
"""
import argparse

from src.config import Config
from src.train import fit


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--fold", type=int, default=None)
    p.add_argument("--run-name", default=None)
    p.add_argument("--num-epochs", type=int, default=None)
    p.add_argument("--image-size", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--limit-per-class", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--init-from", default=None)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    cfg = Config.from_json(args.config)
    if args.fold is not None:
        cfg.val_fold = args.fold
    if args.run_name is not None:
        cfg.run_name = args.run_name
    if args.num_epochs is not None:
        cfg.num_epochs = args.num_epochs
    if args.image_size is not None:
        cfg.image_size = args.image_size
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.limit_per_class is not None:
        cfg.limit_per_class = args.limit_per_class
    if args.num_workers is not None:
        cfg.num_workers = args.num_workers
    if args.seed is not None:
        cfg.seed = args.seed
    if args.device is not None:
        cfg.device = args.device
    if args.init_from is not None:
        cfg.init_from = args.init_from

    fit(cfg, resume=args.resume)


if __name__ == "__main__":
    main()
