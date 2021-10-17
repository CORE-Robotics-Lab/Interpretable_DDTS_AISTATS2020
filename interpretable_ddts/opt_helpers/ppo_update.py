import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np


class PPO:
    def __init__(self, actor_critic_arr, two_nets=True, use_gpu=False):

        lr = 1e-3
        eps = 1e-5
        self.clip_param = 0.2
        self.ppo_epoch = 32
        self.num_mini_batch = 4
        self.value_loss_coef = 0.5
        self.entropy_coef = 0.01
        self.max_grad_norm = 0.5
        self.use_gpu = use_gpu
        if two_nets:
            self.actor = actor_critic_arr[0]
            self.critic = actor_critic_arr[1]
            if self.actor.input_dim > 100:
                self.actor_opt = optim.RMSprop(self.actor.parameters(), lr=1e-5)
                self.critic_opt = optim.RMSprop(self.critic.parameters(), lr=1e-5)
            else:
                self.actor_opt = optim.RMSprop(self.actor.parameters(), lr=1e-2)
                self.critic_opt = optim.RMSprop(self.critic.parameters(), lr=1e-2)
        else:
            self.actor = actor_critic_arr
            self.actor_opt = optim.Adam(self.actor.parameters(), lr=lr, eps=eps)
        self.two_nets = two_nets
        self.epoch_counter = 0

    def batch_updates(self, rollouts, agent_in):
        if self.actor.input_dim < 10:
            batch_size = max(rollouts.step // 16, 1)
            num_iters = rollouts.step // batch_size
        else:
            num_iters = 4
            batch_size = 8
        total_action_loss = torch.Tensor([0])
        total_value_loss = torch.Tensor([0])
        for iteration in range(num_iters):
            total_action_loss = torch.Tensor([0])
            total_value_loss = torch.Tensor([0])
            if self.use_gpu:
                total_action_loss = total_action_loss.cuda()
                total_value_loss = total_value_loss.cuda()

            samples = [rollouts.sample() for _ in range(batch_size)]
            samples = [sample for sample in samples if sample != False]
            if len(samples) <= 0:
                continue
            state = torch.cat([sample['state'][0] for sample in samples], dim=0)
            action_probs = torch.Tensor([sample['action_prob'] for sample in samples])
            adv_targ = torch.Tensor([sample['advantage'] for sample in samples])
            reward = torch.Tensor([sample['reward'] for sample in samples])
            old_action_probs = torch.cat([sample['full_prob_vector'].unsqueeze(0) for sample in samples], dim=0)
            if True in np.array(np.isnan(adv_targ).tolist()) or \
                    True in np.array(np.isnan(reward).tolist()) or \
                    True in np.array(np.isnan(old_action_probs).tolist()):
                continue
            action_taken = torch.Tensor([sample['action_taken'] for sample in samples])
            if self.use_gpu:
                action_taken = action_taken.cuda()
                state = [st.cuda() for st in state]
                action_probs = action_probs.cuda()
                old_action_probs = old_action_probs.cuda()
                adv_targ = adv_targ.cuda()

            new_action_probs = self.actor(state)
            new_value = self.critic(state)

            update_m = Categorical(new_action_probs)
            update_log_probs = update_m.log_prob(action_taken)
            action_indices = [int(action_ind.item()) for action_ind in action_taken]
            new_value = new_value[np.arange(0, len(new_value)), action_indices]
            entropy = update_m.entropy().mean().mul(self.entropy_coef)

            # PPO Updates
            ratio = torch.exp(update_log_probs - action_probs)
            surr1 = ratio * adv_targ
            surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ
            action_loss = -torch.min(surr1, surr2).mean()
            # Policy Gradient:
            # action_loss = (torch.sum(torch.mul(update_log_probs, adv_targ).mul(-1), -1))
            if self.use_gpu:
                reward = reward.cuda()
            value_loss = F.mse_loss(reward, new_value)

            total_value_loss = total_value_loss.add(value_loss)
            total_action_loss = total_action_loss.add(action_loss).sub(entropy)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.critic_opt.zero_grad()
            total_value_loss.backward()
            self.critic_opt.step()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.actor_opt.zero_grad()
            total_action_loss.backward()
            self.actor_opt.step()

        agent_in.reset()
        self.epoch_counter += 1
        return total_action_loss.item(), total_value_loss.item()
