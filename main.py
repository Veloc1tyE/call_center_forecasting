# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 01:02:17 2019

@author: Veloc1ty
"""

# Main script to test the agent and put everything together
# We're actually making it hard for our agent
# Everytime we reset the environment, it's parameters are initialised to
# random, but reasonable values

# Thus, the successful agent has to learn the entire functional space,
# how to optimise it, and must be very general

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from src.CallCentre import CallCentre
from src.FeatureTransformer import FeatureTransformer
from src.Agent import PolicyModel, ValueModel

def optimise_call_td(env, pmodel, vmodel, gamma):
    """
    This is a temporal-difference implementation to 
    optimise a callcentre with an RL agent based on
    the policy gradient method defined in Agent
    """
    observation = env.reset()
    totalReward = 0
    iters = 0
    
    while iters < 100:
        # We don't want this going forever, so just quit after the engine has
        # had enough iterations to optimise
        action = pmodel.sample_action(observation)
        prev_observation = observation
        
        observation, reward = env.step(action)
        # print(observation)
        # Now update the model
        totalReward += reward
        V_next = vmodel.predict(observation)
        G = reward + gamma*V_next
        advantage = G - vmodel.predict(prev_observation)
        pmodel.partial_fit(prev_observation, action, advantage)
        vmodel.partial_fit(prev_observation, G)
        
        iters += 1
        
    return totalReward

def plot_running_avg(totalRewards):
  N = len(totalRewards)
  running_avg = np.empty(N)
  for t in range(N):
    running_avg[t] = totalRewards[max(0, t-100):(t+1)].mean()
  plt.plot(running_avg)
  plt.title("Running Average")
  plt.show()

def main():
    env = CallCentre()
    ft = FeatureTransformer(env, n_components = 100)
    inputs = ft.dimensions
    # Right now our network doesn't have any depth
    pmodel = PolicyModel(ft,inputs, hidden_layer_sizes = [6,4])
    vmodel = ValueModel(ft,inputs, hidden_layer_sizes = [6,4])
    
    # Initialise session
    init = tf.global_variables_initializer()
    session = tf.InteractiveSession()
    session.run(init)
    
    # set session
    pmodel.set_session(session)
    vmodel.set_session(session)
    gamma = 0.95
    
    N = 500
    totalRewards = np.empty(N)
    
    # Now we train!
    for n in range(N):
        # print(n)
        totalReward = optimise_call_td(env, pmodel, vmodel, gamma)
        totalRewards[n] = totalReward
        if n % 1 == 0:
            # Summarise this stage of training
            print("episode:", n, "total reward: %.1f" % totalReward, 
                  "avg reward (last 100): %.1f" % totalRewards[max(0, n-100):(n+1)].mean())
    
    # Final state of model
    print("avg reward for last 100 episodes:", totalRewards[-100:].mean())
    
    plt.plot(totalRewards)
    plt.title("Rewards")
    plt.show()
    
    plot_running_avg(totalRewards)
    
    
if __name__ == '__main__':
    main()
    
    
    
    
    
    