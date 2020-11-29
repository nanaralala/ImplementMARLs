from numpy.lib.function_base import append
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as functional
import gym
import random
from matplotlib import pyplot as plt 

update_iter = 100
memory_capacity = 10000
start_learn_size = 1000
memory_batch_size = 10
epsilon = 0.85
learning_rate = 0.001
gamma = 0.9


class ExperienceReplay(object):
    '''
    ExperienceReplay for agent to learn
    '''
    def __init__(self, size, n_obs):
        self.capacity = size
        self.current_size = 0
        self.buffer = np.zeros((size, n_obs * 2 + 2))
        pass

    def add(self, transition):
        # Transition is (s, a, r, s')
        if self.current_size < self.capacity:
            #print(self.buffer[0].shape)
            #print(transition.shape)
            self.buffer[self.current_size] = transition
            self.current_size += 1
        else:
            ind = random.choice(list(range(self.capacity)))  # Replace randomly
            self.buffer[ind] = transition

    def choose_memory(self, size):
        # Return a list of random chosen transitions
        inds = random.sample(list(range(self.capacity)), size)
        return self.buffer[inds]


class Net(nn.Module):
    '''
    Network structure for target network and action-value network
    '''
    def __init__(self, action_num, state_num):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(state_num, 10)
        self.fc1.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(10, action_num)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = functional.relu(x)
        action_value = self.out(x)
        return action_value


class DQN(object):
    '''
    DQN agent
    '''
    def __init__(self):
        self.target_network = None
        self.action_value_network = None
        self.exp_rep = None
        self.n_action = None
        self.n_observation = None
        self.step_now = 0
        self.optimizer = None
        self.loss_func = None

    def initialize(self, n_obs, n_act):
        self.target_network = Net(n_act, n_obs)
        self.action_value_network = Net(n_act, n_obs)
        self.exp_rep = ExperienceReplay(memory_capacity, n_obs)
        self.n_action = n_act
        self.n_observation = n_obs
        self.step_now = 0
        self.loss_func = nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            self.action_value_network.parameters(), lr=learning_rate)

    def store_to_memory(self, s, a, r, s_new):
        self.exp_rep.add(np.hstack((s, a, r, s_new)))  # Make it a horizon list

    def choose_action(self, observation):
        observation = torch.unsqueeze(
            torch.FloatTensor(observation),
            0)  # Add a dimension to observation, make it a row vector
        if np.random.uniform() < epsilon:  # Greedy
            action_value = self.action_value_network.forward(observation)
            action = torch.max(action_value, 1)[1].data.numpy()[0]
        else:  # Random
            action = np.random.randint(0, self.n_action)
        return action

    def learn(self):
        if self.step_now % update_iter == 0:  # Update target network
            self.target_network.load_state_dict(
                self.action_value_network.state_dict())
        self.step_now += 1
        memory_batch = self.exp_rep.choose_memory(memory_batch_size)
        cur_s = memory_batch[:, :self.
                             n_observation]  # All states(first n values) from each rows
        cur_a = memory_batch[:, self.n_observation:self.n_observation + 1].astype(int)
        cur_r = memory_batch[:, self.n_observation + 1:self.n_observation + 2]
        cur_s_new = memory_batch[:, self.n_observation + 2:]
        cur_s = torch.FloatTensor(cur_s)
        cur_a = torch.LongTensor(cur_a)
        cur_r = torch.FloatTensor(cur_r)
        cur_s_new = torch.FloatTensor(cur_s_new)

        # Select q_eval
        q_eval = self.action_value_network(cur_s).gather(1, cur_a)
        q_next = self.target_network(cur_s_new).detach()
        q_target = cur_r + gamma * q_next.max(1)[0].view(memory_batch_size, 1)
        loss = self.loss_func(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


dqn_agent = DQN()
env = gym.make('CartPole-v0')
env = env.unwrapped
dqn_agent.initialize(env.observation_space.shape[0], env.action_space.n)
rewards=[]
for episode in range(400):
    s = env.reset()
    episode_reward = 0
    while True:
        env.render()
        a = dqn_agent.choose_action(s)  # Choose an action
        s_new, r, done, info = env.step(a)  # Take action
        x, x_dot, theta, theta_dot = s_new
        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians -
              abs(theta)) / env.theta_threshold_radians - 0.5
        reward = r1 + r2
        episode_reward += reward
        dqn_agent.store_to_memory(s, a, reward, s_new)
        if dqn_agent.exp_rep.current_size >= start_learn_size:
            dqn_agent.learn()
        if done:
            print("Episode %d is done now, reward is %f" %
                  (episode + 1, episode_reward))
            rewards.append(episode_reward)
            break
        s = s_new
    

def plot_reward(rewards):
    plt.title("Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.plot(list(range(len(rewards))), rewards)
    plt.show()

plot_reward(rewards)