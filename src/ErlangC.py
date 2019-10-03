# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 21:20:17 2019

@author: Veloc1ty
"""

# Here we define an array of helper functions that
# will later be used to set the reward structure and
# provide feedback to our agent

"""
Example Usage:

Suppose we have 11 agents and a service load of 10 erlangs
And we want to serve each customer in 20 seconds with an
average call taking 180 seconds
        
Then serviceLevel(10,11,20,180) tells us that only 38% of 
customers are being served within the desired time
        
If our desired serviceLevel was 80%, increasing # agents by 3
gives 88.8%, so this is desired
"""

import math

# Basis function
def erlangC(a, n):
    """
    The Erlang C formula gives the probability 
    that a call waits given the traffic intensity
    (a) and the number of agents (n)
    
    Since the formula itself is quite complex, we split
    it up into common terms than regroup
    """
    # Forms the numerator and a normalising 
    # term on the denominator
    basis = (a**n / math.factorial(int(n))) * (n / (n - a))
    # Main factor that affects response probability
    sumAgents = 0
    for i in range(n):
        sumAgents += (a**i / math.factorial(int(i)))

    # From this we can calculate the desired quantity
    return basis / (sumAgents + basis)

def serviceLevel(a, n, target, averageTime):
    """
    This function calculates the proportion of
    customers who are being served within the desired time (target)
    based on the inputs and output of the erlangC function.

    This quantity is often controlled by the call centre,
    and of course it depends on how long employees spend on calls:
        averageTime
        
    This is what our RL agent must optimise
    """
    # First find erlanC output
    erlang = erlangC(a,n)
    # Then factor in targetTime and averageTime
    targetFactor = math.exp(-( (n-a) * (target / averageTime) ))
    # Now the proportion who are being served on time is given by
    return 1 - erlang * targetFactor