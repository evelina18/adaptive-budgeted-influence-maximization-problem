# -*- coding: utf-8 -*-
import numpy as np
import networkx as nx
from .socialNetwork import SocialNetwork


class CTModel(SocialNetwork):
    def __init__(self, env : object, settings : dict):
        """Base function for instantiating the Complex Threshold Model. 
        Derived from the linear threshold model, it considers the influence power and the influence resistance of the nodes, which can change
        along the simulation based on influence decay, influence noise, time form influence.

        Args:
            env (object): environment object
            settings (dict): dictionary with settings parameters
        """
        # save pointer to env
        self.env = env
        self.resistance_noises = settings['influence_model']['resistance_noises']
        self.decays = settings['influence_model']['decays']
        # run super class constructor
        super(CTModel, self).__init__(settings)
        # initialize attributes
        nx.set_node_attributes(self.g, values = 0, name = 'type')
        nx.set_node_attributes(self.g, values = 0.0, name = 'influence')
        nx.set_node_attributes(self.g, values = 0.0, name = 'resistance')
        nx.set_node_attributes(self.g, values = 0, name = 'influence_time')
        # set characteristics
        counter = 0
        n_nodes = len(self.g.nodes)
        node_lst = list(self.g.nodes)
        # for each type of node:
        for i, perc in enumerate(settings['influence_model']['percentage']):
            # for each node in the given subset
            for node in node_lst[counter:(counter + int(n_nodes * perc))]:
                # compute successors
                n_successors = len(list(self.g.successors(node))) / max(self.g.degree())[1]
                # set data
                self.g.nodes[node]['type'] = i
                self.g.nodes[node]['influence'] = 1 - np.exp(- settings['influence_model']['lambda'][i] * n_successors)
                self.g.nodes[node]['resistance'] = np.random.beta(
                    a=settings['influence_model']['a'][i],
                    b=settings['influence_model']['b'][i]
                )
            # update the set of nodes to consider
            counter = counter + int(n_nodes * perc)

    def apply_influence(self, action : list):
        """Simulation of the node influence process starting from choosing a set of nodes from the agent.
        A node becomes active if there exists at least one of the successors which has an influence power greater than its influence resistance.

        Args:
            action (list): graph node set of the chosen node from the agent with get_action() function
        """
        # for all the nodes influenced
        for node in self.state.union(action):
            # for all the successors
            for i in self.g.successors(node):
                # get the resistance of the node
                resistance_noise = self.resistance_noises[self.g.nodes[node]['type']]
                # update the influnce considering the decay
                decay = self.decays[self.g.nodes[node]['type']]
                time_from_influence = self.env.current_time - self.g.nodes[node]['influence_time']
                influence = self.g.nodes[node]['influence'] * (1 - decay * self.g.nodes[node]['status'] ) ** time_from_influence 
                # compute the resistance
                resistance = self.g.nodes[node]['resistance'] + np.random.uniform(
                    low = -resistance_noise,
                    high = resistance_noise,
                )
                # if the influence grater than resistance update the state
                if influence >= resistance:
                    self.g.nodes[i]['status'] = 1
                    self.state.add(i) 
           
            # add node activated in the list of active nodes   
            self.state.add(node)
            if self.g.nodes[node]['influence_time'] == 0:
                self.g.nodes[node]['status']
                self.g.nodes[node]['influence_time'] = self.env.current_time
