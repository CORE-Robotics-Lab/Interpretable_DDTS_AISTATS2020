# Optimization Methods for Interpretable Differentiable Decision Trees in RL

This is the codebase for running DDT agents in OpenAI Gym and the SC2LE (with modifications). The wild-fire domain is not currently available publicly, though it may come in the future.

### Requirements

Requirements are included in the `requirements.txt` file, and this repo itself is a requirement. Install by running the following in the main directory:
```
$ pip install -r requirements.txt
$ pip install -e .
```
Unfortunately, one of the requirements is now out-dated, so you must use the versions specified in the text file. Updating will cause a mismatch between the StarCraft II client and library.

#### StarCraft II
Installing StarCraft II can be a bit of a pain, head to: https://github.com/Blizzard/s2client-proto#downloads to grab the Linux binary from Blizzard. The version that works with this research is 3.16.1, and be sure to install the maps from the DeepMind `pysc2` library: https://github.com/deepmind/pysc2.

### Training DDT or MLP Agents
Training DDT and MLP agents in each environment is relatively straightforward. For the Gym agents (Lunar Lander and Cart Pole), use the `runfiles/gym_runner.py` script. Command-line args allow you to set the agent type and the environment:

* `-a` or `--agent_type`: Which agent to run? String input, use `ddt` or `mlp`. Defaults to `ddt`
* `-e` or `--episodes`: How many episodes to run? Int input, Defaults to 2000.
* `-l` or `--num_leaves`: How many leaves for the DDT/DRL? Int input, defaults to 8
* `-n` or `--num_hidden`: How many hidden layers for the MLP? Int input, defaults to 0
* `-env` or `--env_type`: Which environment to run in? String input, use `cart` or `lunar`. Defaults to `cart`
* `-gpu`: Flag to run on the GPU. Because the GPU isn't really the bottleneck, this isn't a huge speedup.

An example command for a 2-layer MLP on lunar lander for 1000 episodes is:
```
$ python gym_runner.py -a mlp -e 1000 -env lunar -n 2
```

Note that to switch between DRL and DDT, you must manually set the `rule_list` flag on line 109 to True (DRL) or False (DDT).

For the StarCraft II FindAndDefeatZerglings minigame, the commands are much the same, and the `rule_list` flag is on line 249.
```
$ python sc2_minigame_runner.py -a ddt -e 1500 -l 16
```

### Discretizing DDTS:
The `runfiles/run_discrete_agent.py` script is responsible for both finding high-performing discrete policies as well as evaluating all policies. There is often high-variability in the discretized and differentiable policies, so the `run_discrete_agent.py` script helps to search through saved models for the best ones. It is also the script for training sklearn decision trees over expert trajectories.

Command line args include:
* `-d` or `--discretize`: Discretize a DDT? Include for True, otherwise it will just train sklearn decision trees.
* `-env` or `--env_type`: Which environment to use? Options are `FindAndDefeatZerglings`, `cart`, or `lunar`. Defaults to `cart`
* `-m` or `--model_dir`: Where are models stored? Defaults to `../models/`
* `-f` or `--find_model`: Find the best models or use a specific one? Include this flag to search for models.
* `-r` or `--run_model`: Run a model to get statistics for it? Include for True.
* `-n` or `--model_fn`: Filename for a specific model to run. This is optional, leave it out to simply use the best filename from the `--find_model` run.

So, as an example, to search for the best discretized model with on cart pole and run it (assuming the models are saved in `../models/`:
```
$ python run_discrete_agent.py -env cart -f -r -d
```
If instead I want to see which produces the best decision trees from sklearn:
```
$ python run_discrete_agent.py -env cart -f -r
```

For any questions, feel free to contact me as andrew.silva@gatech.edu and the full paper is available [here: Optimization Methods for Interpretable Differentiable Decision Trees in Reinforcement Learning
](https://arxiv.org/abs/1903.09338).
