from gridworld_pa1 import world, plot_Q
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
def choose_action_softmax(Q, state, rg=rg):
    action_probs = softmax(Q[state])
    return rg.choice(len(actions), p=action_probs)

#ep greedy
def choose_action_epsilon(Q, state, epsilon= 0.1, rg=rg):
    if not Q[state[0], state[1]].any():
        return rg.choice(len(actions))
    else:
        if rg.rand() < epsilon:
            return rg.choice(len(actions))
        else:
            return np.argmax(Q[state[0], state[1]])
            
def sarsa(env, Q, alpha = 0.4, gamma=0.9, plot_heat=False, choose_action=choose_action_softmax, episodes = 5000, world_num = -1):
    print_freq = 100
    episode_rewards = np.zeros(episodes)
    steps_to_completion = np.zeros(episodes)
    if plot_heat:
        clear_output(wait=True)
        plot_Q(Q, world_num)
    for ep in tqdm(range(episodes)):
        tot_reward, steps = 0, 0
        
        # Reset environment
        state = env.reset()
        action = choose_action(Q, state)
        done = False
        while not done:
            state_next, reward, done = env.step(state, action)
            action_next = choose_action(Q, state_next)
            
            # Update equation for SARSA
            Q[state, action] += alpha * (reward + gamma * Q[state_next, action_next] - Q[state, action])
            
            tot_reward += reward
            steps += 1
            
            state, action = state_next, action_next

            if state in env.goal_states_seq:
                break
        
        episode_rewards[ep] = tot_reward
        steps_to_completion[ep] = steps
        
        if (ep+1) % print_freq == 0 and plot_heat:
            clear_output(wait=True)
            plot_Q(Q, world_num, message="Episode %d: Reward: %f, Steps: %.2f, Qmax: %.2f, Qmin: %.2f" % (ep+1, np.mean(episode_rewards[ep-print_freq+1:ep]),
                                                                                             np.mean(steps_to_completion[ep-print_freq+1:ep]),
                                                                                             Q.max(), Q.min()))
            
    return Q, episode_rewards, steps_to_completion


            
def qlearning(env, Q, alpha = 0.4, gamma=0.9, plot_heat=False, choose_action=choose_action_softmax, episodes = 5000, world_num = -1):
    print_freq = 100
    episode_rewards = np.zeros(episodes)
    steps_to_completion = np.zeros(episodes)
    if plot_heat:
        clear_output(wait=True)
        plot_Q(Q, world_num)
    for ep in tqdm(range(episodes)):
        tot_reward, steps = 0, 0
        
        # Reset environment
        state = env.reset()
        action = choose_action(Q, state)
        done = False
        while not done:
            state_next, reward, done = env.step(state, action)
            action_next = choose_action(Q, state_next)
            
            # Update equation for SARSA
            Q[state, action] += alpha * (reward + gamma * np.max(Q[state_next]) - Q[state, action])
            
            tot_reward += reward
            steps += 1
            
            state, action = state_next, action_next

            if state in env.goal_states_seq:
                break
        
        episode_rewards[ep] = tot_reward
        steps_to_completion[ep] = steps
        
        if (ep+1) % print_freq == 0 and plot_heat:
            clear_output(wait=True)
            plot_Q(Q,world_num, message="Episode %d: Reward: %f, Steps: %.2f, Qmax: %.2f, Qmin: %.2f" % (ep+1, np.mean(episode_rewards[ep-print_freq+1:ep]),
                                                                                             np.mean(steps_to_completion[ep-print_freq+1:ep]),
                                                                                             Q.max(), Q.min()))
                
    return Q, episode_rewards, steps_to_completion


def plot_solver(env,Q):
    state = env.reset()
    done = False
    steps = 0
    tot_reward = 0

    while not done:
        clear_output(wait=True)
        state, reward, done= env.step(state,Q[state].argmax())
        env.render_world(state)
        steps += 1
        tot_reward += reward
        sleep(0.2)
    print("Steps: %d, Total Reward: %d"%(steps, tot_reward))


def reward_steps(env, world_num, policy, episodes = 5000):
    num_expts = 5
    reward_avgs = []
    steps_avgs =[]

    for i in range(num_expts):
        print("Experiment: %d"%(i+1))
        Q = np.zeros((env.num_states, env.num_actions))

        Q, rewards, steps = policy(env, Q, plot_heat=True, choose_action= choose_action_softmax, world_num=world_num)
        reward_avgs.append(rewards)
        steps_avgs.append(steps)
        
    reward_avgs_mean = np.mean(reward_avgs, axis=0)
    reward_avgs_std = np.std(reward_avgs, axis=0)

    steps_avgs_mean = np.mean(steps_avgs, axis=0)
    steps_avgs_std = np.std(steps_avgs, axis=0)

    plt.figure(figsize=(10, 6))

    # Plot mean
    plt.plot(range(episodes), reward_avgs_mean, label='Reward Avg', color='blue')

    # Plot standard deviation as shaded region
    plt.fill_between(range(episodes), reward_avgs_mean - reward_avgs_std, reward_avgs_mean + reward_avgs_std, alpha=0.3, color='blue')

    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.savefig('world_'+ str(world_num) +'reward_avg.png')
    plt.show()

    plt.figure(figsize=(10, 6))

    # Plot mean
    plt.plot(range(episodes), steps_avgs_mean, label='Steps Avg', color='orange')

    # Plot standard deviation as shaded region
    plt.fill_between(range(episodes), steps_avgs_mean - steps_avgs_std, steps_avgs_mean + steps_avgs_std, alpha=0.3, color='orange')

    plt.xlabel('Episode')
    plt.ylabel('Number of steps to Goal')
    plt.legend()
    plt.savefig('world_'+ str(world_num) +'steps_avg.png')
    plt.show()
