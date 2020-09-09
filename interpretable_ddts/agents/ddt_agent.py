# Created by Andrew Silva on 8/28/19
import torch
from torch.distributions import Categorical
from interpretable_ddts.agents.ddt import DDT
from interpretable_ddts.opt_helpers import replay_buffer, ppo_update
import os
import numpy as np


def save_ddt(fn, model):
    checkpoint = dict()
    mdl_data = dict()
    mdl_data['weights'] = model.layers
    mdl_data['comparators'] = model.comparators
    mdl_data['leaf_init_information'] = model.leaf_init_information
    mdl_data['action_probs'] = model.action_probs
    mdl_data['alpha'] = model.alpha
    mdl_data['input_dim'] = model.input_dim
    mdl_data['is_value'] = model.is_value
    checkpoint['model_data'] = mdl_data
    torch.save(checkpoint, fn)


def load_ddt(fn):
    model_checkpoint = torch.load(fn, map_location='cpu')
    model_data = model_checkpoint['model_data']
    init_weights = [weight.detach().clone().data.cpu().numpy() for weight in model_data['weights']]
    init_comparators = [comp.detach().clone().data.cpu().numpy() for comp in model_data['comparators']]

    new_model = DDT(input_dim=model_data['input_dim'],
                    weights=init_weights,
                    comparators=init_comparators,
                    leaves=model_data['leaf_init_information'],
                    alpha=model_data['alpha'].item(),
                    is_value=model_data['is_value'])
    new_model.action_probs = model_data['action_probs']
    return new_model


def init_rule_list(num_rules, dim_in, dim_out):
    weights = np.random.rand(num_rules, dim_in)
    leaves = []
    comparators = np.random.rand(num_rules, 1)
    for leaf_index in range(num_rules):
        leaves.append([[leaf_index], np.arange(0, leaf_index).tolist(), np.random.rand(dim_out)])
    leaves.append([[], np.arange(0, num_rules).tolist(), np.random.rand(dim_out)])
    return weights, comparators, leaves


class DDTAgent:
    def __init__(self,
                 bot_name='DDT',
                 input_dim=4,
                 output_dim=2,
                 rule_list=False,
                 num_rules=4):
        self.replay_buffer = replay_buffer.ReplayBufferSingleAgent()
        self.bot_name = bot_name
        self.rule_list = rule_list
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.num_rules = num_rules
        if rule_list:
            self.bot_name += str(num_rules)+'_rules'
            init_weights, init_comparators, init_leaves = init_rule_list(num_rules, input_dim, output_dim)
        else:
            init_weights = None
            init_comparators = None
            init_leaves = num_rules
            self.bot_name += str(num_rules) + '_leaves'
        self.action_network = DDT(input_dim=input_dim,
                                  output_dim=output_dim,
                                  weights=init_weights,
                                  comparators=init_comparators,
                                  leaves=init_leaves,
                                  alpha=1,
                                  is_value=False,
                                  use_gpu=False)
        self.value_network = DDT(input_dim=input_dim,
                                 output_dim=output_dim,
                                 weights=init_weights,
                                 comparators=init_comparators,
                                 leaves=init_leaves,
                                 alpha=1,
                                 is_value=True,
                                 use_gpu=False)

        self.ppo = ppo_update.PPO([self.action_network, self.value_network], two_nets=True, use_gpu=False)

        self.last_state = [0, 0, 0, 0]
        self.last_action = 0
        self.last_action_probs = torch.Tensor([0])
        self.last_value_pred = torch.Tensor([[0, 0]])
        self.last_deep_action_probs = None
        self.last_deep_value_pred = [None]*output_dim
        self.full_probs = None
        self.deeper_full_probs = None
        self.reward_history = []
        self.num_steps = 0

    def get_action(self, observation):
        with torch.no_grad():
            obs = torch.Tensor(observation)
            obs = obs.view(1, -1)
            self.last_state = obs

            probs = self.action_network(obs)
            value_pred = self.value_network(obs)
            probs = probs.view(-1).cpu()
            self.full_probs = probs
            if self.action_network.input_dim > 10:
                probs, inds = torch.topk(probs, 3)
            m = Categorical(probs)
            action = m.sample()
            log_probs = m.log_prob(action)
            self.last_action_probs = log_probs.cpu()
            self.last_value_pred = value_pred.view(-1).cpu()

            if self.action_network.input_dim > 10:
                self.last_action = inds[action].cpu()
            else:
                self.last_action = action.cpu()
        if self.action_network.input_dim > 10:
            action = inds[action].item()
        else:
            action = action.item()
        return action

    def save_reward(self, reward):
        self.replay_buffer.insert(obs=[self.last_state],
                                  action_log_probs=self.last_action_probs,
                                  value_preds=self.last_value_pred[self.last_action.item()],
                                  deeper_action_log_probs=self.last_deep_action_probs,
                                  deeper_value_pred=self.last_deep_value_pred[self.last_action.item()],
                                  last_action=self.last_action,
                                  full_probs_vector=self.full_probs,
                                  deeper_full_probs_vector=self.deeper_full_probs,
                                  rewards=reward)
        return True

    def end_episode(self, reward):
        value_loss, action_loss = self.ppo.batch_updates(self.replay_buffer, self)
        self.num_steps += 1
        bot_name = '../txts/' + self.bot_name
        with open(bot_name + '_rewards.txt', 'a') as myfile:
            myfile.write(str(reward) + '\n')

    def reset(self):
        self.replay_buffer.clear()

    def save(self, fn='last'):
        act_fn = fn + self.bot_name + '_actor_' + '.pth.tar'
        val_fn = fn + self.bot_name + '_critic_' + '.pth.tar'

        save_ddt(act_fn, self.action_network)
        save_ddt(val_fn, self.value_network)

    def load(self, fn='last'):
        act_fn = fn + self.bot_name + '_actor_' + '.pth.tar'
        val_fn = fn + self.bot_name + '_critic_' + '.pth.tar'

        if os.path.exists(act_fn):
            self.action_network = load_ddt(act_fn)
            self.value_network = load_ddt(val_fn)

    def __getstate__(self):
        return {
            'action_network': self.action_network,
            'value_network': self.value_network,
            'ppo': self.ppo,
            'bot_name': self.bot_name,
            'rule_list': self.rule_list,
            'output_dim': self.output_dim,
            'input_dim': self.input_dim,
            'num_rules': self.num_rules
        }

    def __setstate__(self, state):
        for key in state:
            setattr(self, key, state[key])

    def duplicate(self):
        new_agent = DDTAgent(bot_name=self.bot_name,
                             input_dim=self.input_dim,
                             output_dim=self.output_dim,
                             rule_list=self.rule_list,
                             num_rules=self.num_rules
                             )
        new_agent.__setstate__(self.__getstate__())
        return new_agent
