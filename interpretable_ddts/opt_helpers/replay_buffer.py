import random
import torch
import numpy as np


class ReplayBufferSingleAgent(object):
    def __init__(self):
        self.states_list = []
        self.action_probs_list = []
        self.value_list = []
        self.hidden_state_list = []
        self.rewards_list = []
        self.deeper_value_list = []
        self.deeper_action_list = []
        self.deeper_advantage_list = []
        self.action_taken_list = []
        self.advantage_list = []
        self.full_probs_list = []
        self.deeper_full_probs_list = []
        self.step = -1

    def __getstate__(self):
        all_data = {
            'states': self.states_list,
            'actions': self.action_probs_list,
            'values': self.value_list,
            'hidden_states': self.hidden_state_list,
            'rewards': self.rewards_list,
            'steps': self.step,
            'deeper_values': self.deeper_value_list,
            'deeper_actions': self.deeper_action_list,
            'actions_taken': self.action_taken_list,
            'advantage_list': self.advantage_list,
            'deeper_advantage_list': self.deeper_advantage_list,
            'full_probs_list': self.full_probs_list,
            'deeper_full_probs_list': self.deeper_full_probs_list
        }
        return all_data

    def __setstate__(self, state):
        self.states_list = state['states']
        self.action_probs_list = state['actions']
        self.value_list = state['values']
        self.hidden_state_list = state['hidden_states']
        self.rewards_list = state['rewards']
        self.step = state['steps']
        self.deeper_value_list = state['deeper_values']
        self.deeper_action_list = state['deeper_actions']
        self.action_taken_list = state['actions_taken']
        self.advantage_list = state['advantage_list']
        self.deeper_advantage_list = state['deeper_advantage_list']
        self.full_probs_list = state['full_probs_list']
        self.deeper_full_probs_list = state['deeper_full_probs_list']

    def extend(self, state):
        self.states_list.extend(state['states'])
        self.action_probs_list.extend(state['actions'])
        self.value_list.extend(state['values'])
        self.hidden_state_list.extend(state['hidden_states'])
        self.rewards_list.extend(state['rewards'])
        self.step += state['steps']
        self.deeper_value_list.extend(state['deeper_values'])
        self.deeper_action_list.extend(state['deeper_actions'])
        self.action_taken_list.extend(state['actions_taken'])
        self.advantage_list.extend(state['advantage_list'])
        self.deeper_advantage_list.extend(state['deeper_advantage_list'])
        self.full_probs_list.extend(state['full_probs_list'])
        self.deeper_full_probs_list.extend(state['deeper_full_probs_list'])

    def insert(self,
               obs=None,
               recurrent_hidden_states=None,
               action_log_probs=None,
               value_preds=None,
               deeper_action_log_probs=None,
               deeper_value_pred=None,
               last_action=None,
               full_probs_vector=None,
               deeper_full_probs_vector=None,
               rewards=None):
        self.states_list.append(obs)
        self.hidden_state_list.append(recurrent_hidden_states)
        self.action_probs_list.append(action_log_probs)
        self.value_list.append(value_preds)
        self.rewards_list.append(rewards)
        self.deeper_action_list.append(deeper_action_log_probs)
        self.deeper_value_list.append(deeper_value_pred)
        self.action_taken_list.append(last_action)
        self.full_probs_list.append(full_probs_vector)
        self.deeper_full_probs_list.append(deeper_full_probs_vector)
        # self.done_list.append(done)
        self.step += 1

    def clear(self):
        del self.states_list[:]
        del self.hidden_state_list[:]
        del self.value_list[:]
        del self.action_probs_list[:]
        del self.rewards_list[:]
        del self.deeper_value_list[:]
        del self.deeper_action_list[:]
        del self.action_taken_list[:]
        del self.deeper_advantage_list[:]
        del self.advantage_list[:]
        del self.full_probs_list[:]
        del self.deeper_full_probs_list[:]
        self.step = 0

    def sample(self):
        # randomly sample a time step
        if len(self.states_list) <= 0:
            return False
        t = random.randint(0, len(self.states_list)-1)
        sample_back = {
            'state': self.states_list[t],
            'hidden_state': self.hidden_state_list[t],
            'action_prob': self.action_probs_list[t],
            'value_pred': self.value_list[t],
            'deeper_action_prob': self.deeper_action_list[t],
            'deeper_value_pred': self.deeper_value_list[t],
            'reward': self.rewards_list[t],
            'advantage': self.advantage_list[t],
            'action_taken': self.action_taken_list[t],
            'deeper_advantage': self.deeper_advantage_list[t],
            'full_prob_vector': self.full_probs_list[t],
            'deeper_full_prob_vector': self.deeper_full_probs_list[t]
        }
        return sample_back


def discount_reward(reward, value, deeper_value):
    R = 0
    rewards = []
    all_rewards = reward
    reward_sum = sum(all_rewards)
    all_values = value
    deeper_all_values = deeper_value
    # Discount future rewards back to the present using gamma
    advantages = []
    deeper_advantages = []

    for r, v, d_v in zip(all_rewards[::-1], all_values[::-1], deeper_all_values[::-1]):
        R = r + 0.99 * R
        rewards.insert(0, R)
        advantages.insert(0, R - v)
        if d_v is not None:
            deeper_advantages.insert(0, R - d_v)
    advantages = torch.Tensor(advantages)
    rewards = torch.Tensor(rewards)

    if len(deeper_advantages) > 0:
        deeper_advantages = torch.Tensor(deeper_advantages)
        deeper_advantages = (deeper_advantages - deeper_advantages.mean()) / (
                deeper_advantages.std() + torch.Tensor([np.finfo(np.float32).eps]))
        deeper_advantage_list = deeper_advantages.detach().clone().cpu().numpy().tolist()
    else:
        deeper_advantage_list = [None] * len(all_rewards)
    # Scale rewards
    # rewards = (rewards - rewards.mean()) / (rewards.std() + torch.Tensor([np.finfo(np.float32).eps]))
    # advantages = (advantages - advantages.mean()) / (advantages.std() + torch.Tensor([np.finfo(np.float32).eps]))
    rewards_list = rewards.detach().clone().cpu().numpy().tolist()
    advantage_list = advantages.detach().clone().cpu().numpy().tolist()
    return rewards_list, advantage_list, deeper_advantage_list
