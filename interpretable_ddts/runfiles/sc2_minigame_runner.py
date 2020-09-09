import sc2
from sc2 import Race
import os
from sc2.constants import *
from sc2.position import Pointlike, Point2
from sc2.player import Bot
import torch
from interpretable_ddts.agents.ddt_agent import DDTAgent

from interpretable_ddts.agents.mlp_agent import MLPAgent
from interpretable_ddts.opt_helpers.replay_buffer import discount_reward
from interpretable_ddts.opt_helpers import sc_helpers
import numpy as np
import torch.multiprocessing as mp
import argparse

DEBUG = False
SUPER_DEBUG = False
if SUPER_DEBUG:
    DEBUG = True

FAILED_REWARD = -0.0
SUCCESS_BUILD_REWARD = 1.
SUCCESS_TRAIN_REWARD = 1.
SUCCESS_SCOUT_REWARD = 1.
SUCCESS_ATTACK_REWARD = 1.
SUCCESS_MINING_REWARD = 1.


class SC2MicroBot(sc2.BotAI):
    def __init__(self, rl_agent, kill_reward=1):
        super(SC2MicroBot, self).__init__()
        self.agent = rl_agent
        self.kill_reward = kill_reward
        self.action_buffer = []
        self.prev_state = None
        self.last_known_enemy_units = []
        self.itercount = 0
        self.last_reward = 0
        self.my_tags = None
        self.agent_list = []
        self.dead_agent_list = []
        self.dead_index_mover = 0
        self.dead_enemies = 0

    async def on_step(self, iteration):

        if iteration == 0:
            self.my_tags = [unit.tag for unit in self.units]
            for unit in self.units:
                self.agent_list.append(self.agent.duplicate())
        else:
            self.last_reward = 0
            for unit in self.state.dead_units:
                if unit in self.my_tags:
                    self.last_reward -= 1
                    self.dead_agent_list.append(self.agent_list[self.my_tags.index(unit)])
                    del self.agent_list[self.my_tags.index(unit)]
                    del self.my_tags[self.my_tags.index(unit)]
                    self.dead_agent_list[-1].save_reward(self.last_reward)
                else:
                    self.last_reward += self.kill_reward
                    self.dead_enemies += 1
            # if len(self.state.dead_units) > 0:
            for agent in self.agent_list:
                agent.save_reward(self.last_reward)
        for unit in self.units:
            if unit.tag not in self.my_tags:
                self.my_tags.append(unit.tag)
                self.agent_list.append(self.agent.duplicate())
        # if iteration % 20 != 0:
        #     return
        all_unit_data = []
        for unit in self.units:
            all_unit_data.append(sc_helpers.get_unit_data(unit))
        while len(all_unit_data) < 3:
            all_unit_data.append([-1, -1, -1, -1])
        for unit, agent in zip(self.units, self.agent_list):
            nearest_allies = sc_helpers.get_nearest_enemies(unit, self.units)
            unit_data = sc_helpers.get_unit_data(unit)
            nearest_enemies = sc_helpers.get_nearest_enemies(unit, self.known_enemy_units)
            unit_data = np.array(unit_data).reshape(-1)
            enemy_data = []
            allied_data = []
            for enemy in nearest_enemies:
                enemy_data.extend(sc_helpers.get_enemy_unit_data(enemy))
            for ally in nearest_allies[1:3]:
                allied_data.extend(sc_helpers.get_unit_data(ally))
            enemy_data = np.array(enemy_data).reshape(-1)
            allied_data = np.array(allied_data).reshape(-1)
            state_in = np.concatenate((unit_data, allied_data, enemy_data))
            action = agent.get_action(state_in)
            await self.execute_unit_action(unit, action, nearest_enemies)
        try:
            await self.do_actions(self.action_buffer)
        except sc2.protocol.ProtocolError:
            print("Not in game?")
            self.action_buffer = []
            return
        self.action_buffer = []

    async def execute_unit_action(self, unit_in, action_in, nearest_enemies):
        if action_in < 4:
            await self.move_unit(unit_in, action_in)
        elif action_in < 9:
            await self.attack_nearest(unit_in, action_in, nearest_enemies)
        else:
            pass

    async def move_unit(self, unit_to_move, direction):
        current_pos = unit_to_move.position
        target_destination = current_pos
        if direction == 0:
            target_destination = [current_pos.x, current_pos.y + 5]
        elif direction == 1:
            target_destination = [current_pos.x + 5, current_pos.y]
        elif direction == 2:
            target_destination = [current_pos.x, current_pos.y - 5]
        elif direction == 3:
            target_destination = [current_pos.x - 5, current_pos.y]
        self.action_buffer.append(unit_to_move.move(Point2(Pointlike(target_destination))))

    async def attack_nearest(self, unit_to_attack, action_in, nearest_enemies_list):
        if len(nearest_enemies_list) > action_in-4:
            target = nearest_enemies_list[action_in-4]
            if target is None:
                return -1
            self.action_buffer.append(unit_to_attack.attack(target))
        else:
            return -1

    def finish_episode(self, game_result):
        print("Game over!")
        if game_result == sc2.Result.Defeat:
            for index in range(len(self.agent_list), 0, -1):
                self.dead_agent_list.append(self.agent_list[index-1])
                self.dead_agent_list[-1].save_reward(-1)
            del self.agent_list[:]
        elif game_result == sc2.Result.Tie:
            reward = 0
        elif game_result == sc2.Result.Victory:
            reward = 0  # - min(self.itercount/500.0, 900) + self.units.amount
        else:
            # ???
            return -13
        if len(self.agent_list) > 0:
            reward_sum = sum(self.agent_list[0].replay_buffer.rewards_list)
        else:
            reward_sum = sum(self.dead_agent_list[-1].replay_buffer.rewards_list)

        for agent_index in range(len(self.agent_list)):
            rewards_list, advantage_list, deeper_advantage_list = discount_reward(
                self.agent_list[agent_index].replay_buffer.rewards_list,
                self.agent_list[agent_index].replay_buffer.value_list,
                self.agent_list[agent_index].replay_buffer.deeper_value_list)
            self.agent_list[agent_index].replay_buffer.rewards_list = rewards_list
            self.agent_list[agent_index].replay_buffer.advantage_list = advantage_list
            self.agent_list[agent_index].replay_buffer.deeper_advantage_list = deeper_advantage_list
        for dead_agent_index in range(len(self.dead_agent_list)):
            rewards_list, advantage_list, deeper_advantage_list = discount_reward(
                self.dead_agent_list[dead_agent_index].replay_buffer.rewards_list,
                self.dead_agent_list[dead_agent_index].replay_buffer.value_list,
                self.dead_agent_list[dead_agent_index].replay_buffer.deeper_value_list)
            self.dead_agent_list[dead_agent_index].replay_buffer.rewards_list = rewards_list
            self.dead_agent_list[dead_agent_index].replay_buffer.advantage_list = advantage_list
            self.dead_agent_list[dead_agent_index].replay_buffer.deeper_advantage_list = deeper_advantage_list
        return self.dead_enemies*self.kill_reward - len(self.dead_agent_list)


def run_episode(q, main_agent, game_mode):
    result = None
    agent_in = main_agent
    kill_reward = 1
    bot = SC2MicroBot(rl_agent=agent_in, kill_reward=kill_reward)

    try:
        result = sc2.run_game(sc2.maps.get(game_mode),
                              [Bot(Race.Terran, bot)],
                              realtime=False)
    except KeyboardInterrupt:
        result = [-1, -1]
    except Exception as e:
        print(str(e))
        print("No worries", e, " carry on please")
    if type(result) == list and len(result) > 1:
        result = result[0]
    reward_sum = bot.finish_episode(result)
    for agent in bot.agent_list+bot.dead_agent_list:
        agent_in.replay_buffer.extend(agent.replay_buffer.__getstate__())
    if q is not None:
        try:
            q.put([reward_sum, agent_in.replay_buffer.__getstate__()])
        except RuntimeError as e:
            print(e)
            return [reward_sum, agent_in.replay_buffer.__getstate__()]
    return [reward_sum, agent_in.replay_buffer.__getstate__()]


def main(episodes, agent, game_mode):
    running_reward_array = []
    # lowered = False
    for episode in range(1, episodes+1):
        successful_runs = 0
        master_reward, reward, running_reward = 0, 0, 0
        try:
            returned_object = run_episode(None, main_agent=agent, game_mode=game_mode)
            master_reward += returned_object[0]
            running_reward_array.append(returned_object[0])
            # agent.replay_buffer.extend(returned_object[1])
            successful_runs += 1
        except MemoryError as e:
            print(e)
            continue
        reward = master_reward / float(successful_runs)
        agent.end_episode(reward)
        running_reward = sum(running_reward_array[-100:]) / float(min(100.0, len(running_reward_array)))
        if episode % 50 == 0:
            print(f'Episode {episode}  Last Reward: {reward}  Average Reward: {running_reward}')
        if episode % 300 == 0:
            agent.save('../models/' + str(episode) + 'th')
    return running_reward_array


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--agent_type", help="architecture of agent to run", type=str, default='ddt')
    parser.add_argument("-env", "--env_type", help="FindAndDefeatZerglings",
                        type=str, default='FindAndDefeatZerglings')
    parser.add_argument("-l", "--num_leaves", help="number of leaves for DDT/DRL ", type=int, default=4)
    parser.add_argument("-n", "--num_hidden", help="number of hidden layers for MLP ", type=int, default=0)
    parser.add_argument("-e", "--episodes", help="how many episodes", type=int, default=1000)
    parser.add_argument("-gpu", help="run on GPU?", action='store_true')

    args = parser.parse_args()
    AGENT_TYPE = args.agent_type  # 'ddt' or 'mlp'
    NUM_EPS = args.episodes  # num episodes Default 1000
    USE_GPU = args.gpu  # Applies for 'prolo' only. use gpu? Default false
    ENV_TYPE = args.env_type
    dim_in = 37
    dim_out = 10
    bot_name = AGENT_TYPE + ENV_TYPE
    mp.set_sharing_strategy('file_system')
    for _ in range(5):

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
        main(episodes=NUM_EPS, agent=policy_agent, game_mode=ENV_TYPE)
