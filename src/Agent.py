# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 23:15:56 2019

@author: Veloc1ty
"""

# This is the my first attempt at solving the ErlangC problem 
# using reinforcement learning

# I assume the reward structure is relatively differentiable 
# (Since the erlangC formula is pretty nice, aside from the division by zero,
#  which I account for in my call-centre class)

# As such, I elected to solve the problem using policy-gradients, 
# parameterized by a neural network

# This is because the action-space is unbounded, and the state-space is 
# effectively infinite, so simple RL techniques (such as what you would use to 
# solve gridworld) will not work in this case

# Also note that in early literature, policy-gradients were optimised using Monte-Carlo methods
# However, this requires a concept of an 'episode', which is not well defined
# for call centre forecasting; therefore, to enable real-world utility
# we optimise using the actor critic method, which allows us to use temporal-difference
# optimisation. This will enable online learning for deployed/training real-world implementations

import sys
import numpy as np
import tensorflow as tf

# We build the hidden layer constructor to enable us
# to test a wide range of architectures
class HiddenLayer:
    def __init__(self, M1, M2, f=tf.nn.elu, use_bias=True):
        self.W = tf.Variable(tf.random_normal(shape=(M1,M2)))
        self.use_bias = use_bias
        if use_bias:
            self.b = tf.Variable(np.zeros(M2).astype(np.float32))
        self.f = f
        
    def forward(self, X):
        if self.use_bias:
            a = tf.matmul(X, self.W) + self.b
        else:
            a = tf.matmul(X, self.W)
        
        return self.f(a)
        # tf.print("activations: ", self.f(a), output_stream=sys.stdout)
    
# Approximates the optimal policy for our agent
class PolicyModel:
    def __init__(self, ft, inputs=5, outputs=3, hidden_layer_sizes = []):
        self.ft = ft
        # Create the graph
        self.hidden_layers = []    
        # Remember that CallCentre has five inputs
        # This can be changed for more complete models,
        # But right now it's just POC
        M1 = inputs
        for M2 in hidden_layer_sizes:
            layer = HiddenLayer(M1, M2)
            self.hidden_layers.append(layer)
            M1 = M2
            
            
        # final layer. We treat outputs as a discrete space, currently either 0,1 or 2
        # This is of course not optimal, and should be improved in future versions
        layer = HiddenLayer(M1, outputs, tf.nn.softmax, use_bias=False)
        self.hidden_layers.append(layer)

        # inputs and targets
        self.X = tf.placeholder(tf.float32, shape=(None, inputs), name='X')
        self.actions = tf.placeholder(tf.int32, shape=(None,), name='actions')
        self.advantages = tf.placeholder(tf.float32, shape=(None,), name='advantages')
        
        # Get final hidden layer
        Z = self.X
        for layer in self.hidden_layers:
            Z = layer.forward(Z)
            
        # calculate output and cost
        p_a_given_s = Z
        self.predict_op = p_a_given_s
        
        selected_probs = tf.log(
                tf.reduce_sum(
                        p_a_given_s * tf.one_hot(self.actions, outputs),
                        reduction_indices=[1]
                        )
                    )
        
        # now calculate cost
        cost = -tf.reduce_sum(self.advantages * selected_probs)
        self.train_op = tf.train.AdamOptimizer(1e-5).minimize(cost)
    
    def set_session(self, session):
        self.session = session
    
    def partial_fit(self, X, actions, advantages):
        # Preprocessing before chucking it through tf
        X = np.atleast_2d(X)
        X = self.ft.transform(X)
        
        actions = np.atleast_1d(actions)
        advantages = np.atleast_1d(advantages)
        self.session.run(
                self.train_op,
                feed_dict={
                        self.X : X,
                        self.actions : actions,
                        self.advantages: advantages
                        }
                    )
        
    def predict(self, X):
        X = np.atleast_2d(X)
        X = self.ft.transform(X)
        return self.session.run(self.predict_op, feed_dict={self.X : X})
    
    def sample_action(self, X):
        p = self.predict(X)[0]
        # print(p)
        return np.random.choice([0,1,2], p=p)
    
    
# Now we create a model for the value function
class ValueModel:
    def __init__(self, ft, inputs=5, hidden_layer_sizes = []):
        self.ft = ft
        self.costs = []
        
        # create the graph
        self.layers = []
        M1 = inputs
        for M2 in hidden_layer_sizes:
            layer = HiddenLayer(M1, M2)
            self.layers.append(layer)
            M1 = M2
            
        # final layer
        layer = HiddenLayer(M1, 1, lambda x: x)
        self.layers.append(layer)
        
        # inputs 
        self.X = tf.placeholder(tf.float32, shape=(None, inputs), name='X')
        # This is our target, the environment reward
        self.Y = tf.placeholder(tf.float32, shape=(None,), name='Y') 
        
        # calculate output and cost
        Z = self.X
        for layer in self.layers:
            Z = layer.forward(Z)
        Y_hat = tf.reshape(Z, [-1]) # the output - this is what our model predicts
        # this is the value our model predicts for some X
        self.predict_op = Y_hat
        
        cost = tf.reduce_sum(tf.square(self.Y - Y_hat))
        self.cost = cost
        # We simply need to train the value function to be as accurate as possible
        self.train_op = tf.train.AdamOptimizer(1e-5).minimize(cost)
        
    def set_session(self,session):
        self.session = session
        
    def partial_fit(self, X, Y):
        # Optimises based on a datapoint
        X = np.atleast_2d(X)
        X = self.ft.transform(X)
        Y = np.atleast_1d(Y)
        
        # Run the training op and optimise
        self.session.run(self.train_op, feed_dict = {self.X : X, self.Y : Y})
        cost = self.session.run(self.cost, feed_dict = {self.X : X, self.Y : Y})
        # We're storing this so we can track agent progress
        self.costs.append(cost)
    
    def predict(self, X):
        X = np.atleast_2d(X)
        X = self.ft.transform(X)
        return self.session.run(self.predict_op, feed_dict={self.X: X})
        
        
        
        
    
