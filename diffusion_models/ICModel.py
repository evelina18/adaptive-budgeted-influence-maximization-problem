# -*- coding: utf-8 -*-
import numpy as np
import networkx as nx
from .socialNetwork import SocialNetwork


class ICModel(SocialNetwork):
    def __init__(self, settings : dict):
        """Base function for instantiating the Independent Cascade Model.
        Each arc in the graph is associated with a probability of influencing.

        Args:
            settings (dict): dictionary with settings parameters
        """
        # run super class constructor
        super(ICModel, self).__init__(settings)

        # for each link define the probability to influence
        for node in self.g.nodes:
            for i in self.g.successors(node):
                self.g[node][i]['probability_to_influence'] = np.random.uniform(
                    low = 0,
                    high = settings['influence_model']['max_influence_prob']
                )
        # set the realized flag
        nx.set_node_attributes(
            self.g, values = False,
            name = "realized"
        )

    def apply_influence(self, action : list):
        """Simulation of the node influence process starting with the selection of a set of nodes from the agent.
        The binomial statistical distribution is used to randomly activate each node with a certain probability.

        Args:
            action (list): graph node set of the chosen node from the agent with get_action() function
        """
        for node in self.state.union(action):
            # if the node has not been influenced
            if not self.g.nodes[node]['realized']:
                # try to influence each successor
                for i in self.g.successors(node):
                    influence_flag = np.random.binomial(
                        n=1,
                        p=self.g[node][i]['probability_to_influence']
                    )
                    if influence_flag == 1:
                        self.g.nodes[i]['status'] = 1
                        self.state.add(i)
                # add node activated in the list of active nodes   
                self.state.add(node)
                # mark node as realized
                self.g.nodes[node]['status'] = 1
                self.g.nodes[node]['realized'] = True
