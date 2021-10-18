#!/usr/bin/env python

import argparse

import wandb


def main():
    args = get_args()

    api = wandb.Api()

    run = api.run(args.run_path[0])
    run.summary["LB"] = args.score[0]
    run.summary.update()

    print(args)


def get_args():
    parser = argparse.ArgumentParser(
        description="""
    Update LB score to wandb dashboard.
    """
    )

    parser.add_argument("--run-path", "-r", nargs=1, required=True, help="Run path")
    parser.add_argument("--score", "-s", nargs=1, required=True, help="Score")

    return parser.parse_args()


if __name__ == "__main__":
    main()
