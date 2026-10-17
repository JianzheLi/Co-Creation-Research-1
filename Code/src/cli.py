import argparse
from pathlib import Path
from .runtime import load_config


def main(stage=None):
    parser = argparse.ArgumentParser(description="DeepRUOT conditional-path experiments")
    parser.add_argument("--config", default="configs/gene.yaml")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--stage", choices=["all", "pairs", "distill", "couplings", "matching", "evaluate"], default=stage or "all")
    args = parser.parse_args()
    config = load_config(args.config)
    run = Path(args.run_dir)
    if args.stage in {"all", "pairs"}:
        from .train_pairs import train_pairs
        train_pairs(config, run)
    if args.stage in {"all", "distill", "couplings"} and config.get("distillation", {}).get("enabled"):
        from .cost_distillation import train_distillation
        train_distillation(config, run)
    if args.stage in {"all", "couplings"}:
        from .solve_couplings import solve_couplings
        if not solve_couplings(config, run):
            from .experiment import plot_results, write_report
            from .runtime import snapshots_for
            plot_results(run, snapshots_for(config, full=True))
            write_report(run, config)
            raise SystemExit("Coupling prerequisites failed. See status.json; downstream training is disabled.")
    if args.stage in {"all", "matching"}:
        from .matching import train_matching
        train_matching(config, run)
    if args.stage in {"all", "evaluate"}:
        from .experiment import generate_and_evaluate
        generate_and_evaluate(config, run)


if __name__ == "__main__":
    main()
