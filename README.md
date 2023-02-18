# Adversarial Malware Generator (AMG)
Adversarial malware generator (AMG) for attacking the GBDT classifier.
## Setup
### Environment
To install all necesarry Python libraries, use the provided Conda environment file *amg-env.yml*.
### PE Files
Place your binary sample into *gym_malware/envs/utils/samples* folder. Make 3 folders there: *train*, *test* and *all* (all samples combined). To switch between validation and final testing phase, change the content of *test* folder

## Train & Testing
For training use the *RAY_train_from_config.py* file. It can be used for example as: ` python RAY_train_from_config.py --agent=DQN --params=params_malware-ember-v0.json --stop-value=50` to train DQN agent with default configuration against GBDT (specified in the `--params` file) for 50 training iterations. All possible parameters are listed in the help guide.

For optimizing hyperparameters use the *tune-hyperparams.py* file. Use as: `python  tune-hyperparams.py --agent=DQN --train-env=malware-train-ember-v0 --stop-value=100`. The hyperparameters to optimize are specified directly in the source code.

For testing use *RAY_test_from_checkpoint* file. Usage: `python RAY_test_from_checkpoint.py --agent=DQN --params={PATH_TO_CONFIG} --checkpoint={PATH_TO_CHECKPOINT} --save-files=True`. 

As configs and checkpoints you can use files from the *BEST_AGENTS* folder. This folder contains config files and checkpoints for each of the tested RL algorithms (DQN, PG, PPO).

