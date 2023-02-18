import argparse
import os
import json
from datetime import datetime

import gym
import gym_malware
from gym_malware.envs.utils import interface

import ray
from ray import air, tune

#import tensorflow as tf
#gpus = tf.config.experimental.list_physical_devices('GPU')
#for gpu in gpus:
#    tf.config.experimental.set_memory_growth(gpu, True)
    
# How to run
# python RAY_train_from_config.py 
# --agent=DQN 
# --params=/home/matous/Documents/CVUT_FIT/1_DP/diploma-thesis/src/AMG/analyze_ray_tune_training/best_params/DQN_malware-train-ember-v0_8ef5c_00000_0_gamma=0.8046,lr=0.0007_2022-11-12_21-28-26/params.json 
# --num-worker=1 --criteria=training_iteration --stop-value=2

parser = argparse.ArgumentParser()
parser.add_argument("--agent", required=True, type=str, help="Name of the agent: [DQN, PPO, etc.].")
parser.add_argument("--name", type=str, default="RAY_train", help="Name of the experiment.")

parser.add_argument("--params", type=str, required=True, help="Path to 'params.json' config.")
parser.add_argument("--checkpoint", type=str, help="Path to checkpoint with trained agent.")

parser.add_argument("--num-cpus", type=int, default=4)
parser.add_argument("--num-gpus", type=int)
parser.add_argument("--num-workers", type=int)

parser.add_argument("--criteria", type=str, default="training_iteration", choices={"timesteps_total", "training_iteration"}) #"time_total_s"
parser.add_argument("--stop-value", type=int, default=1000)


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)

    timelog = (str(datetime.date(datetime.now())) + "_" + str(datetime.time(datetime.now())))

    ray.init(num_cpus=args.num_cpus or None, num_gpus=args.num_gpus or None)

    # Config from file
    config = {}
    with open(args.params) as json_file:
        config = json.load(json_file)

#    config["train_batch_size"] = 128 # For debugging

    # Rewrite params if params provided
    if args.num_gpus:
        config["num_gpus"] = args.num_gpus
    if args.num_workers:
        config["num_workers"] = args.num_workers

    ENV_NAME = config["env"]
    metric = "episode_reward_mean"
    mode = "max"

    # Stop criterion
    stop = {args.criteria: args.stop_value}                   

    RESULTS_DIR = f"RAY_TRAINING/{args.name}_{args.criteria}={args.stop_value}_ray_logs/{ENV_NAME}"
    RESULTS_NAME = f"{args.agent}_{timelog}"

    if not args.checkpoint:
        # Run tune for some iterations and generate checkpoints.
        tuner = tune.Tuner(
            trainable=args.agent,
            param_space=config,
            run_config=air.RunConfig(
                name=RESULTS_NAME,
                local_dir=RESULTS_DIR,
                stop=stop, 
                checkpoint_config=air.CheckpointConfig(
                    num_to_keep=10,
                    checkpoint_score_attribute=metric,
                    checkpoint_score_order=mode,                
                    checkpoint_frequency=1,
                    checkpoint_at_end=True)
            ),
            tune_config=tune.TuneConfig(
                metric=metric,
                mode=mode,
            ),
        )
    else:
        print("restoring agent from checkpoint", args.checkpoint)
        tuner = tune.Tuner.restore(args.checkpoint)

    results = tuner.fit()

    print(results.get_dataframe())
    df = results.get_dataframe(filter_metric=metric, filter_mode=mode)
    df.to_csv(f"{RESULTS_DIR}/{RESULTS_NAME}/results.csv", index = False)

    best_result = results.get_best_result(metric=metric, mode=mode)
    metrics_df = best_result.metrics_dataframe
    print(metrics_df[metric])
    metrics_df.to_csv(f"{RESULTS_DIR}/{RESULTS_NAME}/results_metrics.csv", index = False)


    idx_best_metric = best_result.metrics_dataframe[metric].idxmax()
    value_best_metric = best_result.metrics_dataframe[metric][idx_best_metric]

    print(50*'#')
    print("Best mean {} (over all ""iterations): {}".format(metric, value_best_metric))

    # Confirm that we picked the right trial.
    assert value_best_metric >= results.get_dataframe()[metric].max(), "Wrong checkpoint picked up, not with the highest score"

    print("Provided params {}".format(args.params))
    checkpoint_path = best_result.best_checkpoints[-1][0]._local_path
    print("Best checkpoint {}".format(checkpoint_path))
    value_best_metric = best_result.best_checkpoints[-1][1][metric]
    print("Mean reward of best checkpoint {}".format(value_best_metric))
    
    assert value_best_metric >= results.get_dataframe()[metric].max(), "Wrong checkpoint picked up, not with the highest score"

    os.system(f"cp -rp {checkpoint_path} {RESULTS_DIR}/{RESULTS_NAME}/")
    os.system(f"cp -p {args.params} {RESULTS_DIR}/{RESULTS_NAME}/")


    print(50*'#')










    

