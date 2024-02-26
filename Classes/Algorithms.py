from email import policy
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

 def __init__(self,env,episodes,alpha,gamma,tau,world_num):
        self.env = env
        self.alpha = alpha
        self.tau = tau
        self.gamma = gamma
        self.world_num = world_num
        self.Q = np.zeros((env.num_states, env.num_actions))
        self.state = None
        self.next_state = None
        self.action = None
        self.next_action = None
        self.episodes = episodes



 def choose_action_softmax(self,state,rg=rg):
    a = [0,1,2,3]
    tau = self.tau
    sfm_Q = np.asarray([np.exp(self.Q[state,i]/tau) for i in a])
    sfm_Q = sfm_Q/sum(sfm_Q)
    action = np.random.choice(len(sfm_Q), p = sfm_Q)
    return action
 
 def solve(self,policy,algorithm):
     self.policy = policy
     self.algorithm = algorithm
     self.reset()
     
     self.reward_steps()

 def reset(self):
        self.steps_ = np.zeros(self.env.num_states)
        self.Q = self.Q = np.zeros((self.env.num_states, self.env.num_actions))
        self.state = None
        self.next_state = None
        self.action = None
        self.next_action = None
        
 def performance_plots(self,world_num,episodes,reward_avgs_mean,reward_avgs_std,steps_avgs_mean,steps_avgs_std):
        
        plt.figure(figsize=(10, 6))
        plt.plot(range(episodes), reward_avgs_mean, label='Reward Avg', color='blue')
        plt.fill_between(range(episodes), reward_avgs_mean - reward_avgs_std, reward_avgs_mean + reward_avgs_std, alpha=0.3, color='blue')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.legend()
        plt.savefig('world_'+ str(world_num)+"_"+self.algorithm.__name__ +'_reward_avg.png')
        plt.show()
        plt.figure(figsize=(10, 6))
        plt.plot(range(episodes), steps_avgs_mean, label='Steps Avg', color='orange')
        
# Plot standard deviation as shaded region
        plt.fill_between(range(episodes), steps_avgs_mean - steps_avgs_std, steps_avgs_mean + steps_avgs_std, alpha=0.3, color='orange')
        plt.xlabel('Episode')
        plt.ylabel('Number of steps to Goal')
        plt.legend()
        plt.savefig('world_'+ str(world_num)+"_"+self.algorithm.__name__ +'_ steps_avg.png')
        self.steps_ = np.round(self.steps_/5,2)
        x = np.reshape(self.steps_, (10, 10))

        plt.figure(figsize=(10, 10))
    # Plot the array with numbers on each cell
        plt.imshow(x, cmap='viridis', interpolation='nearest')
# Add numbers on each cell
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                plt.text(j, i, x[i, j], ha='center', va='center', color='black')
                
        plt.savefig('world_' + str(self.world_num)+"_"+self.algorithm.__name__ +'_Visit_count.png')     
           
        ''' Q = np.flipud(self.Q_avgs.reshape(10,10,4))
        plt.figure(figsize=(10,10))
        plt.title("Q plot")
        plt.pcolor(Q.max(-1), edgecolors='k', linewidths=2)
        plt.colorbar()
        def x_direct(a):
            if a in [UP, DOWN]:
                return 0
            return 1 if a == RIGHT else -1
        def y_direct(a):
            if a in [RIGHT, LEFT]:
                return 0
            return 1 if a == UP else -1
        policy = Q.argmax(-1)
        policyx = np.vectorize(x_direct)(policy)
        policyy = np.vectorize(y_direct)(policy)
        idx = np.indices(policy.shape)
        plt.quiver(idx[1].ravel()+0.5, idx[0].ravel()+0.5, policyx.ravel(), policyy.ravel(), pivot="middle", color='red')
        plt.savefig('world_' + str(self.world_num) + '_Q_plot.png')
        plt.show() '''
        
    
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
        plot_Q(self.Q, world_num,self.algorithm)
    for ep in tqdm(range(self.episodes)):
        tot_reward, steps = 0, 0
        
        # Reset environment
        self.state = self.env.reset()
        self.action = choose_action(self.state)
        done = False
        while not done:
            self.next_state, reward, done = self.env.step(self.state, self.action)
            self.steps_[self.next_state] += 1
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
            plot_Q(self.Q, world_num,self.algorithm, message="Episode %d: Reward: %f, Steps: %.2f, Qmax: %.2f, Qmin: %.2f" % (ep+1, np.mean(episode_rewards[ep-print_freq+1:ep]),
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
        plot_Q(self.Q, world_num,self.algorithm)
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
    num_expts = 1
    reward_avgs = []
    steps_avgs =[]
    Q_avg = np.zeros((self.env.num_states, self.env.num_actions))

    for i in range(num_expts):
        print("Experiment: %d"%(i+1))
        Q = np.zeros((self.env.num_states, self.env.num_actions))
        Q, rewards, steps = self.algorithm(plot_heat=True)
        Q_avg = np.append(Q_avg,Q)
        reward_avgs.append(rewards)
        steps_avgs.append(steps)
        
    reward_avgs_mean = np.mean(reward_avgs, axis=0)
    self.Q_avgs = np.mean(Q_avg, axis=0)
    reward_avgs_std = np.std(reward_avgs, axis=0)
    steps_avgs_mean = np.mean(steps_avgs, axis=0)
    steps_avgs_std = np.std(steps_avgs, axis=0)
    self.performance_plots(self.world_num,self.episodes,reward_avgs_mean,reward_avgs_std,steps_avgs_mean,steps_avgs_std)