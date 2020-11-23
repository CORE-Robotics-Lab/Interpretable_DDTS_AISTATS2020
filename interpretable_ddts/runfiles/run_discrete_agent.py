# Created by Andrew Silva on 5/10/19
import torch
import numpy as np
import os
from interpretable_ddts.opt_helpers.discretization import convert_to_discrete
from interpretable_ddts.agents.ddt_agent import load_ddt, DDTAgent
from interpretable_ddts.agents.ddt import DDT
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from interpretable_ddts.opt_helpers.sklearn_to_ddt import ddt_init_from_dt
import matplotlib.pyplot as plt
from interpretable_ddts.runfiles.sc2_minigame_runner import run_episode as micro_episode
from interpretable_ddts.runfiles.gym_runner import run_episode as gym_episode


def search_for_good_model(env):
    # Be sure to comment out gym_runner.gym_episode env.render
    max_reward = -9999
    max_std = -9999
    max_fuzzy_reward = -9999
    max_fuzzy_std = -9999
    best_fn = 'non'
    best_fuzzy_fn = 'non'
    num_runs = 15
    all_results = []
    for fn in os.listdir(model_dir):

        if env in fn and 'actor' in fn and 'ddt' in fn:
            print(f"FN = {fn}")
            final_deep_actor_fn = os.path.join(model_dir, fn)
            fda = load_ddt(final_deep_actor_fn)

            policy_agent = DDTAgent(bot_name='crispytester',
                                    input_dim=37,
                                    output_dim=10)

            policy_agent.action_network = fda
            policy_agent.value_network = fda
            reward_after_five = []
            master_states = []
            for _ in range(15):
                if env == "FindAndDefeatZerglings":
                    try:
                        reward, replay_buffer = micro_episode(None, policy_agent, env)
                    except Exception as e:
                        continue
                if env in ['cart', 'lunar']:
                    reward, replay_buffer = gym_episode(None, policy_agent, env)
                master_states.extend(replay_buffer['states'])
                reward_after_five.append(reward)

            crispy_actor = convert_to_discrete(policy_agent.action_network, master_states)

            policy_agent.action_network = crispy_actor

            crispy_reward = []
            for _ in range(num_runs):
                if env == "FindAndDefeatZerglings":
                    try:
                        crispy_out, replay_buffer = micro_episode(None, policy_agent, env)
                    except Exception as e:
                        crispy_out = -3
                        continue
                if env in ['cart', 'lunar']:
                    crispy_out, replay_buffer = gym_episode(None, policy_agent, env)

                crispy_reward.append(crispy_out)
            if np.mean(reward_after_five) > max_fuzzy_reward:
                max_fuzzy_reward = np.mean(reward_after_five)
                max_fuzzy_std = np.std(reward_after_five)
                best_fuzzy_fn = fn
            if np.mean(crispy_reward) > max_reward:
                max_reward = np.mean(crispy_reward)
                max_std = np.std(crispy_reward)
                best_fn = fn
            all_results.append([fn, np.mean(reward_after_five), np.std(reward_after_five), np.mean(crispy_reward), np.std(crispy_reward)])
            print(fn)
            print(f"Average reward after 5 runs is {np.mean(reward_after_five)}")
            print(f"Average reward for the crispy network after {num_runs} runs is {np.mean(crispy_reward)}")
    return best_fuzzy_fn, best_fn, max_fuzzy_reward, max_fuzzy_std, max_reward, max_std, all_results


def run_a_model(fn, args_in, seed=None):
    num_runs = 15
    if 'cart' in fn:
        env = 'cart'
    elif 'lunar' in fn:
        env = 'lunar'
    elif 'FindAndDefeatZerglings' in fn:
        env = 'FindAndDefeatZerglings'
    final_deep_actor_fn = os.path.join(model_dir, fn)
    final_deep_critic_fn = os.path.join(model_dir, fn)

    fda = load_ddt(final_deep_actor_fn)
    fdc = load_ddt(final_deep_critic_fn)

    policy_agent = DDTAgent(bot_name='crispytester',
                            input_dim=37,
                            output_dim=10)

    # fda.comparators.data = fda.comparators.data.unsqueeze(-1)
    policy_agent.action_network = fda
    # fsc.comparators.data = fsc.comparators.data.unsqueeze(-1)
    policy_agent.value_network = fdc
    master_states = []
    master_actions = []
    reward_after_five = 0
    for _ in range(num_runs):
        if env == 'FindAndDefeatZerglings':
            reward, replay_buffer = micro_episode(None, policy_agent, game_mode=env)
        if env in ['cart', 'lunar']:
            reward, replay_buffer = gym_episode(None, policy_agent, env)
        master_states.extend(replay_buffer['states'])
        master_actions.extend(replay_buffer['actions_taken'])
        reward_after_five += reward
    print(f"Average reward after {num_runs} runs is {reward_after_five/num_runs}")

    master_states = torch.cat([state[0] for state in master_states], dim=0)
    if args_in.discretize:
        crispy_actor = convert_to_discrete(policy_agent.action_network)  # Discretize DDT
    else:
        ###### test with a DT #######
        x_train = [state.cpu().numpy().reshape(-1) for state in master_states]
        y_train = [action.cpu().numpy().reshape(-1) for action in master_actions]
        clf = DecisionTreeClassifier(max_depth=3)
        clf.fit(x_train, y_train)
        plt.figure(figsize=(20, 20))
        plot_tree(clf, filled=True)
        plt.savefig('tree.png')
        init_weights, init_comparators, init_leaves = ddt_init_from_dt(clf)
        crispy_actor = DDT(input_dim=len(x_train[0]),
                           output_dim=len(np.unique(y_train)),
                           weights=init_weights,
                           comparators=init_comparators,
                           leaves=init_leaves,
                           alpha=99999.,
                           is_value=False,
                           use_gpu=False)

    policy_agent.action_network = crispy_actor
    crispy_reward = []
    for i in range(num_runs):
        if env == 'FindAndDefeatZerglings':
            crispy_out, replay_buffer = micro_episode(None, policy_agent, game_mode=env)
        if env in ['cart', 'lunar']:
            torch.random.manual_seed(seed + i)
            np.random.seed(seed + i)
            crispy_out, replay_buffer = gym_episode(None, policy_agent, env, seed=seed+i)
        crispy_reward.append(crispy_out)

    leaves = crispy_actor.leaf_init_information
    for leaf_ind in range(len(leaves)):
        leaves[leaf_ind][-1] = np.argmax(leaves[leaf_ind][-1])
    print(leaves)
    print(crispy_actor.comparators.detach().numpy().reshape(-1))
    ddt_weights = crispy_actor.layers.detach().numpy()
    print(np.argmax(np.abs(ddt_weights), axis=1))

    print(f"Average reward after {num_runs} runs is {reward_after_five/num_runs}")

    print(f"Average reward for the crispy network after {num_runs} runs is {np.mean(crispy_reward)} with std {np.std(crispy_reward)}")


def fc_state_dict(fn=''):
    fc_model = torch.load(fn)
    print(fc_model['actor'])


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--discretize", help="train sklearn tree or discretize ddt?", action='store_true')
    parser.add_argument("-env", "--env_type", help="FindAndDefeatZerglings, cart, or lunar", type=str, default="cart")
    parser.add_argument("-m", "--model_dir", help="where are models stored?", default="../models", type=str)
    parser.add_argument('-f', '--find_model', help="find the best models?", action="store_true")
    parser.add_argument('-r', '--run_model', help="run a model?", action="store_true")
    parser.add_argument('-n', '--model_fn', help="model filename for running", type=str, default="")
    args = parser.parse_args()

    envir = args.env_type
    model_dir = args.model_dir
    # args.run_model = True
    # args.discretize = True

    if args.find_model:
        diff_fn, discrete_fn, diff_reward, diff_std, disc_reward, disc_std, all_results = search_for_good_model(envir)
        print(f"Best differentiable file: {diff_fn} with {diff_reward} reward and {diff_std} std")
        print(f"Best discrete file: {discrete_fn} with {disc_reward} reward and {disc_std} std")
    if args.run_model:
        if len(args.model_fn) < 1:
            discrete_fn = (os.path.join(model_dir, discrete_fn))
        else:
            discrete_fn = args.model_fn
        run_a_model(discrete_fn, args, seed=12496)
        # cartpole random seeds include: [11421, 12494, 12495, 12496,
        # 30867, 30868, 30869, 30870, 30871, 30872, 34662, 38979, 38980, 45603, 45604, 45605, 45606, 46760, 46761,
        # 50266, 50267, 54857, 65926, 70614, 79986, 79987, 79988, 79989]
