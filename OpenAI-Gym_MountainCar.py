#!/usr/bin/python
print(
'''
                      .__    .__                   
  _____ _____    ____ |  |__ |__| ____   ____      
 /     \\__  \ _/ ___\|  |  \|  |/    \_/ __ \     
|  Y Y  \/ __ \\  \___|   Y  \  |   |  \  ___/     
|__|_|  (____  /\___  >___|  /__|___|  /\___  >    
      \/     \/     \/     \/        \/     \/     
.__machine learning with A7MD0V..__                
|  |   ____ _____ _______  ____ |__| ____    ____  
|  | _/ __ \\__  \\_  __ \/    \|  |/    \  / ___\ 
|  |_\  ___/ / __ \|  | \/   |  \  |   |  \/ /_/  >
|____/\___  >____  /__|  |___|  /__|___|  /\___  / 
          \/     \/           \/        \//_____/
using TensorFlow 1.40 and Python 3.5 ----2017-->

72 65 69 6E 66 6F 72 63 65 6D 65 6E 74 
6C 65 61 72 6E 69 6E 67           
'''

'''
Using Reinforcement Learning: Q-Learning to play MountainCar from OpenAI's Gym

Agent: Car
Goal: Climb mountain on right
Actions: Left, Right, Nothing

Performance: Climb moutain ASAP
Environment: Mountain Valley
Actuators: Two Wheels
Sensors: Velocity and Position of car

'''
# ----------=: import libraries :=-----------------------------------------
import gym
from gym import wrappers
import numpy as np
# ----------=: define parameters/variables :=------------------------------
# here we define the options and tweak the knobs of our agent/environment

n_states = 40
iter_max = 10000

# Learning rate
start_lr = 1.0 
min_lr = 0.003
gamma = 1.0
t_max = 10000
eps = 0.02
# ----------=: define  :=-------------------------------------------
def run_episode(env, policy=None, render=False):
    obs = env.reset()
    total_reward = 0
    step_idx = 0
    for _ in range(t_max):
        if render:
            env.render()
        if policy is None:
            action = env.action_space.sample()
        else:
            a,b = obs_to_state(env, obs)
            action = policy[a][b]
        obs, reward, done, _ = env.step(action)
        total_reward += gamma ** step_idx * reward
        step_idx += 1
        if done:
            break
    return total_reward

def obs_to_state(env, obs):
    """ Maps an observation to state """
	# Create an MDP containing states
    env_low = env.observation_space.low
    env_high = env.observation_space.high
    env_dx = (env_high - env_low) / n_states
    a = int((obs[0] - env_low[0])/env_dx[0])
    b = int((obs[1] - env_low[1])/env_dx[1])
    return a, b
	
# ----------=: Main function :=-------------------------------------------
if __name__ == '__main__':
    env_name = 'MountainCar-v0'
    env = gym.make(env_name)
    env.seed(0)
    np.random.seed(0)
    print ('----- Decision Making using Q Learning -----')
    q_table = np.zeros((n_states, n_states, 3))
    for i in range(iter_max):
        obs = env.reset()
        total_reward = 0
        ## eta: learning rate is decreased at each step
        eta = max(min_lr, start_lr * (0.85 ** (i//100)))
        for j in range(t_max):
            a, b = obs_to_state(env, obs)
            if np.random.uniform(0, 1) < eps:
                action = np.random.choice(env.action_space.n)
            else:
                logits = q_table[a][b]
                logits_exp = np.exp(logits)
                probs = logits_exp / np.sum(logits_exp)
                action = np.random.choice(env.action_space.n, p=probs)
            obs, reward, done, _ = env.step(action)
            total_reward += reward
#------------------ update q table
            a_, b_ = obs_to_state(env, obs)
            q_table[a][b][action] = q_table[a][b][action] + eta * (reward + gamma *  np.max(q_table[a_][b_]) - q_table[a][b][action])
            if done:
                break
        if i % 100 == 0:
            print('Iteration #%d -- Total reward = %d.' %(i+1, total_reward))
    solution_policy = np.argmax(q_table, axis=2)
    solution_policy_scores = [run_episode(env, solution_policy, False) for _ in range(100)]
    print("Average score of solution = ", np.mean(solution_policy_scores))
	
    # Visualize the environment
    run_episode(env, solution_policy, True)
	#-----------------------------------------------------------------------
	
	# credits to Moustafa Alzantot and Siraj Raval