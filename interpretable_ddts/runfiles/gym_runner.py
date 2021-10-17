# Created by Andrew Silva on 8/28/19
import gym
import numpy as np
import torch
from interpretable_ddts.agents.ddt_agent import DDTAgent
from interpretable_ddts.agents.mlp_agent import MLPAgent
from interpretable_ddts.opt_helpers.replay_buffer import discount_reward
import torch.multiprocessing as mp
import argparse
import copy
import random


def run_episode(q, agent_in, ENV_NAME, seed=0):
    agent = agent_in.duplicate()
    if ENV_NAME == 'lunar':
        env = gym.make('LunarLander-v2')
    elif ENV_NAME == 'cart':
        env = gym.make('CartPole-v1')
    else:
        raise Exception('No valid environment selected')
    done = False
    torch.manual_seed(seed)
    env.seed(seed)
    np.random.seed(seed)
    env.action_space.seed(seed)
    random.seed(seed)
    state = env.reset()  # Reset environment and record the starting state

    while not done:
        action = agent.get_action(state)
        # Step through environment using chosen action
        state, reward, done, _ = env.step(action)
        # env.render()
        # Save reward
        agent.save_reward(reward)
        if done:
            break
    reward_sum = np.sum(agent.replay_buffer.rewards_list)
    rewards_list, advantage_list, deeper_advantage_list = discount_reward(agent.replay_buffer.rewards_list,
                                                                          agent.replay_buffer.value_list,
                                                                          agent.replay_buffer.deeper_value_list)
    agent.replay_buffer.rewards_list = rewards_list
    agent.replay_buffer.advantage_list = advantage_list
    agent.replay_buffer.deeper_advantage_list = deeper_advantage_list

    to_return = [reward_sum, copy.deepcopy(agent.replay_buffer.__getstate__())]
    if q is not None:
        try:
            q.put(to_return)
        except RuntimeError as e:
            print(e)
            return to_return
    return to_return


def main(episodes, agent, ENV_NAME):
    running_reward_array = []
    for episode in range(episodes):
        reward = 0
        returned_object = run_episode(None, agent_in=agent, ENV_NAME=ENV_NAME)
        reward += returned_object[0]
        running_reward_array.append(returned_object[0])
        agent.replay_buffer.extend(returned_object[1])
        if reward >= 499:
            agent.save('../models/'+str(episode)+'th')
        agent.end_episode(reward)

        running_reward = sum(running_reward_array[-100:]) / float(min(100.0, len(running_reward_array)))
        if episode % 50 == 0:
            print(f'Episode {episode}  Last Reward: {reward}  Average Reward: {running_reward}')
        if episode % 500 == 0:
            agent.save('../models/'+str(episode)+'th')

    return running_reward_array


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--agent_type", help="architecture of agent to run", type=str, default='ddt')
    parser.add_argument("-e", "--episodes", help="how many episodes", type=int, default=2000)
    parser.add_argument("-l", "--num_leaves", help="number of leaves for DDT/DRL ", type=int, default=8)
    parser.add_argument("-n", "--num_hidden", help="number of hidden layers for MLP ", type=int, default=0)
    parser.add_argument("-env", "--env_type", help="environment to run on", type=str, default='cart')
    parser.add_argument("-gpu", help="run on GPU?", action='store_true')

    args = parser.parse_args()
    AGENT_TYPE = args.agent_type  # 'ddt', 'mlp'
    NUM_EPS = args.episodes  # num episodes Default 1000
    ENV_TYPE = args.env_type  # 'cart' or 'lunar' Default 'cart'
    USE_GPU = args.gpu  # Applies for 'prolo' only. use gpu? Default false
    if ENV_TYPE == 'lunar':
        init_env = gym.make('LunarLander-v2')
        dim_in = init_env.observation_space.shape[0]
        dim_out = init_env.action_space.n
    elif ENV_TYPE == 'cart':
        init_env = gym.make('CartPole-v1')
        dim_in = init_env.observation_space.shape[0]
        dim_out = init_env.action_space.n
    else:
        raise Exception('No valid environment selected')

    print(f"Agent {AGENT_TYPE} on {ENV_TYPE} ")
    # mp.set_start_method('spawn')
    mp.set_sharing_strategy('file_system')
    for i in range(5):
        bot_name = AGENT_TYPE + ENV_TYPE
        if USE_GPU:
            bot_name += 'GPU'
        if AGENT_TYPE == 'ddt':
            policy_agent = DDTAgent(bot_name=bot_name,
                                    input_dim=dim_in,
                                    output_dim=dim_out,
                                    rule_list=False,
                                    num_rules=args.num_leaves)
        elif AGENT_TYPE == 'mlp':
            policy_agent = MLPAgent(input_dim=dim_in,
                                    bot_name=bot_name,
                                    output_dim=dim_out,
                                    num_hidden=args.num_hidden)
        else:
            raise Exception('No valid network selected')
        reward_array = main(NUM_EPS, policy_agent, ENV_TYPE)
