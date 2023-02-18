import argparse

import gym
import gym_malware

print("imported gym-malware...")
from ray import tune, init

print("imported ray")
import numpy as np
import glob
import os
from datetime import datetime

# How to run
 #python tune-hyperparams.py 
 # --agent=DQN 
 # --train-env=malware-train-ember-v0 
 # --name=tune-hyperparams_iter=50_obs=2048 
 # --num-gpus=0 --criteria=training_iteration --stop-value=50
 

parser = argparse.ArgumentParser(description = "Using Ray tune to search for optimal hyperparameters.")
parser.add_argument("--train-env", required = True, type = str, help = "Name of the train gym-malware environment. (must be registered)")
parser.add_argument("--agent", required = True, type = str, help = "Name of the agent: [DQN, PPO, etc.].")
parser.add_argument("--name", type=str, default="tune-hyperparams", help="Name of the experiment.")
parser.add_argument("--num-samples", default = 1, type = int, help = "How many times should the experiment repeat itself.")
parser.add_argument("--criteria", type=str, default="training_iteration", choices={"timesteps_total", "training_iteration"}) #"time_total_s"
parser.add_argument("--stop-value", type=int, default=100)

parser.add_argument("--num-cpus", type=int, default=10)
parser.add_argument("--num-gpus", type=int, default=0)
parser.add_argument("--num-workers", type=int, default=0)

parser.add_argument(
    "--framework",
    choices=["tf", "tf2", "tfe", "torch"],
    default="tf2",
    help="The DL framework specifier.",
)


# TODO clean results before running


def main(args: argparse.Namespace) -> None:
    print("Starting arguments:", args)
   
    gym_malware_config = gym_malware.get_config(args.train_env)
    assert gym_malware_config, f"Couldn't find {args.traing_env} configuration in gym_malware."

    print("Found gym_malware config:", gym_malware_config)
    ENV_NAME = gym_malware_config["name"]

    timelog = (str(datetime.date(datetime.now())) + "_" + str(datetime.time(datetime.now())))

    init(num_cpus=args.num_cpus, num_gpus=args.num_gpus)
    #init(num_cpus=4, num_gpus=0)

   # num_gpus = 0 # Local worker
   # num_gpus_per_worker = 0 # Remote worker
   # if args.num_gpus > 0: 
   #     #num_gpus = 0.01
   #     #num_gpus_per_worker = (args.num_gpus - num_gpus) / args.num_workers
   #     num_gpus = args.num_gpus

    RESULTS_DIR = f"RAY_TRAINING/{args.name}_{args.criteria}={args.stop_value}_ray_logs/{ENV_NAME}"
    RESULTS_NAME = f"{timelog}_{args.agent}"

    results = tune.run(
        name=RESULTS_NAME,
        local_dir=RESULTS_DIR,
        run_or_experiment = [args.agent],
        stop={args.criteria: args.stop_value},
        num_samples = args.num_samples, # Number of repeats
        
        metric = "episode_reward_mean",
        mode = "max",        
        config={
            "env": ENV_NAME,
            "gamma" : tune.grid_search([0.99, 0.75, 0.5]),
            "lr": tune.grid_search([0.01, 0.001, 0.0001]),
            "framework": args.framework,
            "eager_tracing": args.framework in ["tfe", "tf2"], # Run with tracing enabled for tfe/tf2

           # "num_gpus": num_gpus,
           # "num_gpus_per_worker": num_gpus_per_worker,
            "num_workers": args.num_workers,
        
        }
    )
        
    print("Best config is", results.get_best_config(metric="episode_reward_mean", mode = "max"))

    df = results.dataframe()
    df.to_csv(f"{RESULTS_DIR}/{RESULTS_NAME}/{ENV_NAME}.csv", index = False)

    print(f"tune-hyperparams.py with args {args} successfuly done.")


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)

    main(args)
