import argparse
import os
import json
from datetime import datetime
import random

import gym
import gym_malware
from gym_malware.envs.utils import interface

import ray
from ray import air, tune
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.algorithms.dqn import DQN
from ray.rllib.algorithms.a2c import A2C
from ray.rllib.algorithms.es import ES
from ray.rllib.algorithms.marwil import MARWIL
from ray.rllib.algorithms.pg import PG

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# How to run
# python RAY_test_from_checkpoint.py 
# --agent=DQN 
# --checkpoint=RAY_train_DQN_tmp_ray_logs/malware-train-ember-v0/2022-11-16_11\:03\:32.634304_DQN/DQN_malware-train-ember-v0_f1ee1_00000_0_2022-11-16_11-03-40/checkpoint_000003/checkpoint-3 
# --params=analyze_ray_tune_training/best_params/malware-train-ember-v0/DQN/params_3.json 
# --num-gpus=1 --num-workers=1 --save-files=True

parser = argparse.ArgumentParser()
parser.add_argument("--agent", required=True, type=str, help="Name of the agent: [DQN, PPO, etc.].")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint with trained agent.")
parser.add_argument("--params", type=str, required=True, help="Path to 'params.json' config.")
parser.add_argument("--name", type=str, default="RAY_test", help="Name of the experiment.")
parser.add_argument("--save-files", type=bool, default=False, help="Indicate wheter to save adversarial samples.")

parser.add_argument("--num-cpus", type=int, default=1)
parser.add_argument("--num-gpus", type=int, default=0)
parser.add_argument("--num-workers", type=int, default=0)

if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    checkpoint_path = args.checkpoint

    config = {}
    with open(args.params) as json_file:
        config = json.load(json_file)

    print("Provided config:", config)

    # Rewrite params if params provided
    if args.num_gpus:
        config["num_gpus"] = args.num_gpus
    if args.num_workers:
        config["num_workers"] = args.num_workers

    config["num_gpus_per_worker"] = 0 #args.num_gpus / args.num_workers    

    TEST_ENV = config['env'].replace("train", "test")
    config['env'] = TEST_ENV

    ray.init(num_cpus=args.num_cpus or None, num_gpus=args.num_gpus or None)

    # Load trained agent
    if args.agent == "PPO":
        agent = PPO(config=config)
    elif args.agent == "DQN":
        config['explore'] = False # Do not set for Policy Gradient Algorithms
        agent = DQN(config=config)
    elif args.agent == "A2C":
        agent = A2C(config=config)
    elif args.agent == "ES":
        config['explore'] = False # Do not set for Policy Gradient Algorithms
        agent = ES(config=config)
    elif args.agent == "MARWIL":
        agent = MARWIL(config=config)
    elif args.agent == "PG":
        agent = PG(config=config)
    else:
        assert False, f"RLLib agent {args.agent} not known."

    agent.restore(checkpoint_path)
    

    timelog = (str(datetime.date(datetime.now())) + "_" + str(datetime.time(datetime.now())))
    RESULTS_DIR = f"RAY_TESTING/{args.name}_ray_logs/{TEST_ENV}"
    RESULTS_NAME = f"{timelog}_{args.agent}"
 
    # Testing        
    env = gym.make(TEST_ENV)

    # Create evaded folder
    save_folder = os.path.join(f"{RESULTS_DIR}/{RESULTS_NAME}/mod_binaries")
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    else:  # Remove old files
        interface.remove_files(save_folder) 

    # Prepare results dictionary
    results_dictionary = {
        "env": TEST_ENV,
        "algo": args.agent,
        "config": config,
        "checkpoint_path": checkpoint_path,
        "results": []
    }    

    MALWARE_LABEL = 1.0
    BENIGN_LABEL = 0.0
    total = len(env.available_sha256_list)
    num_test_samples = 0
    num_evaded = 0
    num_skipped = 0

    print(f"Testing on {total} binary files.")

#    print(env.available_sha256_list)

    for i in range(total):

        #if i >= 10: # debugging
        #    break
        
        print(100*'#')
        sha256 = env.available_sha256_list[i]
        bytez = interface.fetch_file(sha256)
        label = env.label_function(bytez)
        original_size = len(bytez)#interface.get_size_sha256(sha256)
        #output_path = os.path.join(save_folder, sha256)

        print(f"File {i + 1}/{total} - {sha256}") 
        
        if label != interface.true_label(sha256): # TODO !!!! CHANGE !!!!
            print(f"File {sha256} already missclassified, skipping...")
            results_dictionary["results"].append({
                "file_name": sha256,
                "evaded": True,
                "size_before": original_size,
                "size_after": None,
                "actions": [],
                "reward": None
            })

            num_skipped += 1

            if args.save_files:
                with open(os.path.join(save_folder, f"{sha256}_skip"), 'wb') as outfile:
                    outfile.write(bytez)

            continue
            
        num_test_samples += 1
        episode_reward = 0
        done = False
        info = {}
        obs = env.reset(sha256=sha256) # Setup environment to 'sha256' file (cannot be already misclassified by the target classifier)
        prev_obs = None

        while not done:
            prev_obs = obs
            action = agent.compute_single_action(obs)
            #action = random.randint(0, 9) # RANDOM agent
            obs, reward, done, info = env.step(action)
            episode_reward += reward

            if (prev_obs==obs).all():
                print(f"Observation space DID-NOT change -- action: {action}")
                                    
        if info['evaded']:
            num_evaded += 1

        if args.save_files:
            with open( os.path.join(save_folder, f'{sha256}{"_evasive" if info["evaded"] else "_fail"}'), 'wb') as outfile:
                outfile.write(info['bytez'])

        new_size = len(info['bytez']) #interface.get_size(output_path)
        # assert original_size <= new_size, "modified file is smaller than the original file (current modifications should not decrease size of the file)"

        if original_size > new_size:
            print(f"original_size={original_size}, new_size={new_size}")
            print("modified file is smaller than the original file")

        # Save result    
        results_dictionary["results"].append({
            "file_name": sha256,
            "evaded": info["evaded"],
            "size_before": original_size,
            "size_after": new_size,
            "actions": info["actions"],
            "reward": episode_reward
        })


    print(100*'#')
    print(f"Test results ({TEST_ENV}): {num_evaded}/{num_test_samples} = {num_evaded/num_test_samples}, skipped={num_skipped}")

    #file1 = open(os.path.join(f"{RESULTS_DIR}/{RESULTS_NAME}/", f"{args.name}_{args.agent}_{config['gamma']}_{config['lr']}_summary.json"), 'w')
    file1 = open(os.path.join(f"{RESULTS_DIR}/{RESULTS_NAME}/", f"{args.name}_{TEST_ENV}_{args.agent}_summary.json"), 'w')
    file1.write(json.dumps(results_dictionary, indent=4))

    file1.close()    
    print(f"Results are saved in {RESULTS_DIR}/{RESULTS_NAME}")



    

