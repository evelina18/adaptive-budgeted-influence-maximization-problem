# -*- coding: utf-8 -*-
import numpy as np
from agents.agent import Agent
from envs.dynamicSocialNetwork import DynamicSocialNetwork


class DummyAgent(Agent):

    def __init__(self, env: DynamicSocialNetwork, settings : dict):
        """Base function for instantiating the dummy agent 

        Args:
            env (DynamicSocialNetwork): environment in which the agent acts
            settings (dict): dictionary with settings parameters
        """
        super(DummyAgent, self).__init__(env, settings)
        # set pointer to env
        self.env = env

    def get_action(self, state : np.array):
        """Choose a node from the current state of the graph, based on random selection

        Args:
            state (np.array): graph nodes set of the influenced nodes

        Returns:
            list: random influenced node selected
        """
        nodes_state, budget = state
        # pick at random one node not already influenced
        set_influenced_nodes = set(nodes_state)
        set_not_influenced_nodes = set(list(self.env.social_net.g.nodes())) - set_influenced_nodes
        # pick random node:
        if len(set_not_influenced_nodes) == 0:
            return []
        random_node = np.random.choice(list(set_not_influenced_nodes))
        # return node if budget is enough
        if budget < self.env.social_net.g.nodes[random_node]['cost']:
            return []
        else:
            return [random_node]

    def learn(self, epochs : int = 1000):
        """Function to learn how to make optimized choices in the get action function

        Args:
            epochs (int, optional): number of tests done by the agent during the learning process. Defaults to 1000.
        """
        pass

