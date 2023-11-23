# -*- coding: utf-8 -*-
import numpy as np
import networkx as nx
from .socialNetwork import SocialNetwork


class LTModel(SocialNetwork):
    def __init__(self, settings : dict):
        """Base function for instantiating the Linear Threshold Model.

        Args:
            settings (dict): dictionary with settings parameters
        """
        # run super class constructor
        super(LTModel, self).__init__(settings)
        
        # initialize attribute
        nx.set_node_attributes(
            self.g, values = 0,
            name = "influence_resistence"
        )
        for node in self.g.nodes:
            # set influence_resistence:
            self.g.nodes[node]['influence_resistence'] = np.random.uniform(0,1)
            # get the number of successors 
            for i in self.g.successors(node):
                self.g[node][i]['influence_power'] = np.random.uniform(
                    low = 0,
                    high = settings['influence_model']['max_influence_power']
                )
    
    def apply_influence(self, action : list):
        """Simulation of the node influence process starting from choosing a set of nodes from the agent.
        A node becomes active if a weighted sum of the active successor nodes is greater than its resistance.

        Args:
            action (list): graph node set of the chosen nodes from the agent with get_action() function
        """
        active_nodes = self.state.union(action)
        dict_res = {}
        for node in active_nodes:
            # Add node in the active list
            self.g.nodes[node]['status'] = 1
            self.state.add(node)
            # compute the total influence on each node
            for j in self.g.successors(node):
                if j in dict_res:
                    dict_res[j] += self.g[node][j]["influence_power"]
                else:
                    dict_res[j] = self.g[node][j]["influence_power"]
        # Update the status of the node if the influence_resistence is exceed
        for key, ele in dict_res.items():
            if ele > self.g.nodes[j]["influence_resistence"]:
                self.g.nodes[key]['status'] = 1
                self.state.add(key)
