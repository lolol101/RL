from collections import defaultdict

import os
import sys
import numpy as np

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

from sumo_rl import SumoEnvironment
from sumo_rl.agents import QLAgent
from sumo_rl.exploration import EpsilonGreedy

class MyQLAgent:
    def __init__(self, starting_state, state_space, action_space, alpha=0.1, gamma=0.99, exploration_strategy=None):
        self.state = starting_state
        self.state_space = state_space
        self.action_space = action_space
        self.alpha = alpha
        self.gamma = gamma

        self.Q1 = defaultdict(lambda: np.zeros(self.action_space.n, dtype=float))
        self.Q2 = defaultdict(lambda: np.zeros(self.action_space.n, dtype=float))

        self.exploration = exploration_strategy

        self.action = None
        self.acc_reward = 0.0

    def act(self):
        combined_q = self.Q1[self.state] + self.Q2[self.state]

        temp_table = {self.state: combined_q}
        self.action = self.exploration.choose(temp_table, self.state, self.action_space)
        return self.action

    def learn(self, next_state, reward, done=False):
        if np.random.random() < 0.5:
            best_a = int(np.argmax(self.Q1[next_state]))
            target = reward if done else reward + self.gamma * self.Q2[next_state][best_a]
            td_error = target - self.Q1[self.state][self.action]
            self.Q1[self.state][self.action] += self.alpha * td_error
        else:
            best_a = int(np.argmax(self.Q2[next_state]))
            target = reward if done else reward + self.gamma * self.Q1[next_state][best_a]
            td_error = target - self.Q2[self.state][self.action]
            self.Q2[self.state][self.action] += self.alpha * td_error

        self.acc_reward += reward
        self.state = next_state

if __name__ == "__main__":
    alpha = 0.1
    gamma = 0.99
    decay = 0.995
    runs = 3
    episodes = 4

    env = SumoEnvironment(
        net_file="./sumo_rl/nets/4x4-Lucas/4x4.net.xml",
        route_file="./sumo_rl/nets/4x4-Lucas/4x4c1c2c1c2.rou.xml",
        use_gui=False,
        num_seconds=10000,
        min_green=5,
        delta_time=5,
    )

    for run in range(1, runs + 1):
        initial_states = env.reset()
        ql_agents = {
            ts: MyQLAgent(
                starting_state=env.encode(initial_states[ts], ts),
                state_space=env.observation_space,
                action_space=env.action_space,
                alpha=alpha,
                gamma=gamma,
                exploration_strategy=EpsilonGreedy(initial_epsilon=0.05, min_epsilon=0.005, decay=decay),
            )
            for ts in env.ts_ids
        }

        for episode in range(1, episodes + 1):
            if episode != 1:
                initial_states = env.reset()
                for ts in initial_states.keys():
                    ql_agents[ts].state = env.encode(initial_states[ts], ts)

            infos = []
            done = {"__all__": False}
            while not done["__all__"]:
                actions = {ts: ql_agents[ts].act() for ts in ql_agents.keys()}

                s, r, done, info = env.step(action=actions)

                for agent_id in s.keys():
                    ql_agents[agent_id].learn(next_state=env.encode(s[agent_id], agent_id), reward=r[agent_id])

            env.save_csv(f"outputs/4x4/ql-4x4grid_run_hw1{run}", episode)

    env.close()
