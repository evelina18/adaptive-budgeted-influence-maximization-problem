# -*- coding: utf-8 -*-
import gym
import numpy as np
from diffusion_models import ICModel, CTModel, LTModel


class DynamicSocialNetwork(gym.Env):

    def __init__(self, settings : dict):
        """Base function for instantiating the influence model, 
        characterizing the dynamic social network 

        Args:
            settings (dict): dictionary with settings parameters

        Raises:
            ValueError: invalid influence_model
        """
        super(DynamicSocialNetwork, self).__init__()
        # set data from settings
        self.time_horizon = settings['time_horizon']
        self.g_name = settings['graph_type']
        self.initial_budget = settings['budget']
        self.budget = settings['budget']
        self.l = settings['lambda']
        # create the Social Network
        if settings['influence_model']['type'] == 'LTM':
            self.social_net = LTModel(
                settings
            )
        elif settings['influence_model']['type'] == 'ICM':
            self.social_net = ICModel(
                settings
            )
        elif settings['influence_model']['type'] == 'CTM': 
            self.social_net = CTModel(
                self, settings
            )
        else:
            raise ValueError('invalid influence_model')
        # Initialize quantities
        self.current_time = 0
        self.total_reward = 0
        self.old_n_nodes_influenced = 0


    def reset(self):
        """Function to reset the status attribute of the nodes and the budget to the initial one

        Returns:
            list: a set of nodes
            float: initial budget

        """
        self.current_time = 0
        for node in self.social_net.g.nodes:
            self.social_net.g.nodes[node]['status'] = 0
        self.budget = self.initial_budget
        return (self.social_net.state, self.budget)
    
    def set_state(self, state : np.array):
        """Function to set the attribute status of the nodes

        Args:
            state (np.array): graph nodes set of the influenced nodes
        """
        self.social_net.set_state(state)

    def step(self, action : list):
        """Advances the state of the environment given the action from the agent(s)
        Returns the state of the environment, the reward value, whether the simulation is done, current time

        Args:
            action (list): graph node set of the chosen nodes from the agent with get_action() function

        Returns:
            list: new set of influenced nodes
            float: reward obtained from the action chosen 
            bool: set to true if the current time is equal to the time horizon
            dict: {'current_time': current_time} 
        """
        # compute reward
        action_cost = self.social_net.get_cost(action)
        new_influenced = self.social_net.get_n_influenced() - self.old_n_nodes_influenced
        reward = self.l * new_influenced - action_cost
        # update old_n_nodes_influenced
        self.old_n_nodes_influenced = self.social_net.get_n_influenced()
        # run influence:
        self.social_net.apply_influence(action)
        # update budget
        self.budget = self.budget - self.social_net.get_cost(action)
        # define new state
        new_state = (self.social_net.state, self.budget)
        # update time
        self.current_time += 1
        info = {
            'current_time': self.current_time
        }
        done = self.time_horizon == self.current_time
        return new_state, reward, done, info

    def plot(self):
        """Plotting the social network graph, coloring influenced nodes
        """
        self.social_net.plot()
