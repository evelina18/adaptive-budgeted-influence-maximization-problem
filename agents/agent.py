# -*- coding: utf-8 -*-
import abc
import numpy as np
from abc import abstractmethod
from envs import DynamicSocialNetwork


class Agent(object):
    """Base class for agent"""

    __metaclass__ = abc.ABCMeta

    @abstractmethod
    def __init__(self, env : DynamicSocialNetwork, settings : dict):
        """Base function for instantiating the agent 

        Args:
            env (DynamicSocialNetwork): environment in which the agent acts
            settings (dict): dictionary with settings parameters
        """
        self.name = None

    @abstractmethod
    def get_action(self, state : np.array):
        """Choose a node from the current state of the graph

        Args:
            state (np.array): graph nodes set of the influenced nodes
        """
        pass
    
    @abstractmethod
    def learn(self, epochs = 1000):
        """Function to learn how to make optimized choosing in the get action function

        Args:
            epochs (int, optional): number of tests done by the agent during the learning process. Defaults to 1000.
        """
        pass

    