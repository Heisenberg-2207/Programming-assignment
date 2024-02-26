from Gridworld import world, plot_Q
import numpy as np
from IPython.display import clear_output
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.special import softmax
from time import sleep

#same actions are used for plotting the Q value graph
DOWN = 1
UP = 0
LEFT = 2
RIGHT = 3
actions = [DOWN, UP, LEFT, RIGHT]

seed = 42
rg = np.random.RandomState(seed)

# Softmax

class Solver():

 def __init__(self,env,episodes,alpha,gamma,world_num):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.world_num = world_num
        self.Q = np.zeros((env.num_states, env.num_actions))
        self.state = None
        self.next_state = None
        self.action = None
        self.next_action = None
        self.episodes = episodes



 def choose_action_softmax(self,state, tau = 1, rg=rg):
        action_probs = softmax(self.Q[state]/tau)
        return rg.choice(len(actions), p=action_probs)
 
 def solve(self,policy,algorithm):
     self.policy = policy
     self.algorithm = algorithm
     self.reset()
     self.sarsa()
     self.plot_solver()

 def reset(self):
        self.Q = self.Q = np.zeros((self.env.num_states, self.env.num_actions))
        self.state = None
        self.next_state = None
        self.action = None
        self.next_action = None

#ep greedy
 def choose_action_epsilon(self,state, epsilon= 0.1, rg=rg):
    if not self.Q[state[0], state[1]].any():
        return rg.choice(len(actions))
    else:
        if rg.rand() < epsilon:
            return rg.choice(len(actions))
        else:
            return np.argmax(self.Q[state[0], state[1]])
            
 def sarsa(self, plot_heat=True):
    print_freq = 100
    
    choose_action = self.policy
    world_num = self.world_num
    episode_rewards = np.zeros(self.episodes)
    steps_to_completion = np.zeros(self.episodes)
    if plot_heat:
        clear_output(wait=True)
        plot_Q(self.Q, world_num)
    for ep in tqdm(range(self.episodes)):
        tot_reward, steps = 0, 0
        
        # Reset environment
        self.state = self.env.reset()
        self.action = choose_action(self.state)
        done = False
        while not done:
            self.next_state, reward, done = self.env.step(self.state, self.action)
            self.next_action = choose_action(self.next_state)
            
            # Update equation for SARSA
            self.Q[self.state, self.action] += self.alpha * (reward + self.gamma * self.Q[self.next_state, self.next_action] - self.Q[self.state, self.action])
            
            tot_reward += reward
            steps += 1
            
            self.state, self.action = self.next_state, self.next_action

            if self.state in self.env.goal_states_seq:
                break
        
        episode_rewards[ep] = tot_reward
        steps_to_completion[ep] = steps
        
        if (ep+1) % print_freq == 0 and plot_heat:
            clear_output(wait=True)
            plot_Q(self.Q, world_num, message="Episode %d: Reward: %f, Steps: %.2f, Qmax: %.2f, Qmin: %.2f" % (ep+1, np.mean(episode_rewards[ep-print_freq+1:ep]),
                                                                                             np.mean(steps_to_completion[ep-print_freq+1:ep]),
                                                                                             self.Q.max(), self.Q.min()))
            
    return self.Q, episode_rewards, steps_to_completion


            
 def qlearning(self, plot_heat=False):
    print_freq = 100
    choose_action = self.policy
    world_num = self.world_num
    episode_rewards = np.zeros(self.episodes)
    steps_to_completion = np.zeros(self.episodes)
    if plot_heat:
        clear_output(wait=True)
        plot_Q(self.Q, world_num)
    for ep in tqdm(range(self.episodes)):
        tot_reward, steps = 0, 0
        
        # Reset environment
        self.state = self.env.reset()
        self.action = choose_action(self.state)
        done = False
        while not done:
            self.next_state, reward, done = self.env.step(self.state, self.action)
            self.next_action = choose_action(self.next_state)
            
            # Update equation for SARSA
            self.Q[self.state, self.action] += self.alpha * (reward + self.gamma * np.max(self.Q[self.next_state]) - self.Q[self.state, self.action])
            
            tot_reward += reward
            steps += 1
            
            self.state, self.action = self.next_state, self.next_action

            if self.state in self.env.goal_states_seq:
                break
        
        episode_rewards[ep] = tot_reward
        steps_to_completion[ep] = steps
        
        if (ep+1) % print_freq == 0 and plot_heat:
            clear_output(wait=True)
            plot_Q(self.Q, world_num, message="Episode %d: Reward: %f, Steps: %.2f, Qmax: %.2f, Qmin: %.2f" % (ep+1, np.mean(episode_rewards[ep-print_freq+1:ep]),
                                                                                             np.mean(steps_to_completion[ep-print_freq+1:ep]),
                                                                                             self.Q.max(), self.Q.min()))
            
    return self.Q, episode_rewards, steps_to_completion


 def plot_solver(self):
    state = self.env.reset()
    done = False
    steps = 0
    tot_reward = 0

    while not done:
        clear_output(wait=True)
        state, reward, done= self.env.step(state,self.Q[state].argmax())
        self.env.render_world(state)
        steps += 1
        tot_reward += reward
        sleep(0.2)
    print("Steps: %d, Total Reward: %d"%(steps, tot_reward))


 def reward_steps(self):
    num_expts = 5
    reward_avgs = []
    steps_avgs =[]

    for i in range(num_expts):
        print("Experiment: %d"%(i+1))
        Q = np.zeros((self.env.num_states, self.env.num_actions))
        Q, rewards, steps = self.policy(plot_heat=True)
        reward_avgs.append(rewards)
        steps_avgs.append(steps)
        
    reward_avgs_mean = np.mean(reward_avgs, axis=0)
    reward_avgs_std = np.std(reward_avgs, axis=0)
    steps_avgs_mean = np.mean(steps_avgs, axis=0)
    steps_avgs_std = np.std(steps_avgs, axis=0)
    self.env.performance_plots(self.episodes,reward_avgs_mean,reward_avgs_std,steps_avgs_mean,steps_avgs_std)
