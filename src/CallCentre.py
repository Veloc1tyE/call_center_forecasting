# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 22:32:21 2019

@author: Veloc1ty
"""

# This is a very, very simple RL environment
# Essentially, all it does is call functions 
# from the ErlangC library.

# This enables it to inform the agent about whether
# it is reaching/meeting its desired efficiency

# The agent obtains a reward for exceeding the desired
# proportion of customers who are serviced within the target time 
# by the minimal amount possible, thus ensuring that the call centre
# allocates the optimal amount of resources.

# In all other cases the reward structure incentivises an
# increase/decrease in the number of workers 
# until this goal has been reached, the reward is smooth and differentiable
# to encourage stability and effective temporal-difference learning


import math
import numpy as np
import random
from ErlangC import serviceLevel

class CallCentre:
    def __init__(self, call_volume = 10, num_workers = 11, 
                 target_service_time = 20, 
                 average_required_time = 180,
                 target_service_proportion = 0.8):
        """
        Provides all necessary parameters to use the ErlangC
        functions.
        
        call_volume is the service load in Erlangs (call-hours)
        
        num_workers is the current number of workers who are
        servicing this call-load
        
        target_service_time is the time the call-centre aims to service
        its customers in (they should not be on hold for longer)
        
        average_required_time is how long it takes to service a customer
        on average
        
        target_service_proportion is the customer proportion who will be
        served within the target time, note this is the optimisation goal
        of our agent
        """
        self.a = call_volume
        self.n = num_workers
        self.target = target_service_time
        self.average = average_required_time
        self.goal = target_service_proportion
        
    def step(self, amount):
        """
        Adjusts the number of workers that are currently servicing,
        this is the interface by which the RL agent attemps to 
        optimise its goal.
        
        We assume that the agent only has control over the number of 
        workers in service, the interface can, however, be easily adjusted
        to give the agent control over a multidimensional space of parameters
        in future versions.
        """
        # First we ensure self.n remains valid
        if self.n + amount <= 0:
            # Do nothing and move on
            return self.getState()
        # Note that in the Erlang C formula, num_workers cannot be the
        # same as call volume due to division by zero
        if self.n + amount == self.a:
            # This effectively does nothing
            return self.getState()
        # we pass the checks
        self.n += amount
        return self.getState()
    
    def getReward(self):
        """
        This is the main feedback mechanism for the RL agent:
            
        We give the agent a positive reward for accomplishing its goal;
        otherwise, we give it a negative reward to encourage it to 
        reach its goal as fast as possible
        
        Done variable informs the game when it is finished
        """
        
        
        # Now this it the main section for calculating reward
        # First, calculate the proportion who are serviced on time
        currentProportion = serviceLevel(self.a, self.n, 
                                         self.target, self.average)
        # print("reward: ")
        # print(currentProportion)
        # Now we want so see how this compares to our goal
        if currentProportion >= self.goal:
            # This is great! 
            # Now we just need to check everything works out
            if self.n - 1 != self.a:
                # We only need to decrement n by one
                # And check that it is not above the threshold
                return self.checkValid(1, currentProportion)
            else:
                # We need to decrement n by two
                # because decrementing by one would cause
                # division by zero (issue handling)
                return self.checkValid(2, currentProportion)             
        else:
            # Goal has not been reached, so reward is lower
            # Note that simply using erlangC output as reward would
            # cause instability, hence the exp
            return min(math.exp(currentProportion), self.goal)

    def checkValid(self, decrement, currentProportion):
        # Simple helper method to assist in calculating reward
        belowProportion = serviceLevel(self.a, self.n - decrement,
                                               self.target, self.average)
        if belowProportion >= self.goal:
            # This isn't the minimum amount possible
            # We penalise for unnecessary resource use, while still
            # incentivising the required level of service (better over than under)
            return math.exp(currentProportion) - 0.05*self.n
        else:
            # Yay! The agent has optimised correctly!
            # This incentivizes bare minimum resource use
            return 10
        
    def sample(self):
        """ 
        Here we set reasonable bounds for what parameters we would expect
        This is useful because the RL agent must be general, and deal
        well with a range of situations, so sampling allows us to do this,
        as well as define an input space for the feature transformer
        
        Note that these bounds are very conservative due to too-large bounds
        causing overflow errors in the ErlangC formula
        """
        # Expect call volume to be between 1 and 30
        call_volume = random.randint(1,30)
        # Expect num_workers to be between 1 and 50
        num_workers = random.randint(1,50)
        # Expect target time to be between 10 and 30 seconds
        target_required_time = random.randint(10,30)
        # Expect average_required_time to be betwen 60 and 560 seconds
        average_required_time = random.randint(60,560)
        # Expect target_service_proportion to be between 0.5 and 0.99
        target_service_proportion = random.uniform(0.5, 0.99)
        # Now return an array
        return np.array([call_volume, num_workers,
                target_required_time,
                average_required_time, 
                target_service_proportion])
        
    def reset(self):
        # Sets random initial values for all the parameters
        self.a = random.randint(2,30)
        self.n = max(1, self.a + random.randint(-30,-1))
        self.target = random.randint(10,30)
        self.average = random.randint(60,560)
        self.goal = random.uniform(0.5,0.99)
        return np.array([self.a, self.n, self.target, 
                self.average, self.goal])
        
    def getObservation(self):
        # provides the call-centres internal state
        return np.array([self.a, self.n, self.target, 
                self.average, self.goal])
        
    def getState(self):
        # Helper function to unpack relevant variables
        observation = self.getObservation()
        reward = self.getReward()
        
        return observation, reward
        
        
        
        
        

