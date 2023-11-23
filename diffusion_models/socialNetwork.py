# -*- coding: utf-8 -*-
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


class SocialNetwork():

    def __init__(self, settings : dict):
        """Base function for instantiating the social network graph

        Args:
            settings (dict): dictionary with settings parameters

        Raises:
            Exception: Influence value not valid
        """
        # get data from settings
        graph_settings = settings['graph_type']
        influence_type = settings['influence_model']['type']
        cost_settings = settings["cost_type"]
        seed_settings = settings["seed"]

        # Check if the influence is in the specified types
        if influence_type not in ['LTM', 'ICM', 'CTM']:
            raise Exception("Influence value not valid")
        # save important data
        self.influence_type = influence_type
        # initialize state: it contains the indexes of all the active nodes
        self.state = set()
        # generate graph
        self.generate_graph(graph_settings, cost_settings, seed_settings)

    def generate_graph(self, graph_settings : dict, cost_settings : dict, seed_settings : int):
        """Function that generates the graph based on input parameters

        Args:
            graph_settings (dict): dictionary with graph settings parameters
            cost_settings (dict): dictionary with cost settings parameters
            seed_settings (int): seed value

        Raises:
            ValueError: Graph name not know
        """
        if graph_settings['name'] =='scale_free_graph':
            self.g = nx.generators.directed.scale_free_graph(
                graph_settings['n_nodes'], 
                graph_settings['alfa'],
                graph_settings['beta'], 
                graph_settings['gamma'],  
                graph_settings['delta_in'],  
                graph_settings['delta_out'],
                seed=seed_settings
            )
        elif graph_settings['name'] =='erdos_renyi':
            self.g = nx.erdos_renyi_graph(
                graph_settings['n_nodes'],
                graph_settings['p'],
                directed=True,
                seed=seed_settings
            )
        elif graph_settings['name'] =='watt_strogatz':
            self.g = nx.watts_strogatz_graph(
                graph_settings['n_nodes'],
                int(graph_settings['n_nodes']/graph_settings['n_neigh']),
                graph_settings['edge_prob'],
                seed=seed_settings
            )
        elif graph_settings['name'] =='regular_random':
            self.g = nx.random_regular_graph(
                graph_settings['node_degree'],
                graph_settings['n_nodes'],
                seed=seed_settings
            )
        elif graph_settings['name'] =='barabasi_albert':
            self.g = nx.barabasi_albert_graph(
                n=graph_settings['n_nodes'], 
                m=graph_settings['stub'],
                seed=seed_settings
            )
        elif graph_settings['name'] =='power':
            self.g = nx.powerlaw_cluster_graph(
                graph_settings['n_nodes'], 
                graph_settings['stubs'],
                graph_settings['prob'],
                seed=seed_settings
            )
        elif graph_settings['name'] =='random_lobster':
            self.g = nx.random_lobster(
                graph_settings['n_nodes'], 
                graph_settings['p1'],
                graph_settings['p2'],
                seed=seed_settings
            )
        elif graph_settings['name'] =='real_graph':
            self.g = nx.read_edgelist(
                graph_settings['file_path'],
                nodetype=int,
                create_using=nx.DiGraph()
            )
        else:
            raise ValueError('Graph name not know')

        # transforming in a directed graph if needed
        if isinstance(self.g, nx.Graph):
            self.g = nx.DiGraph(
                nx.to_directed(self.g)
            )

        # the cost of a node is equal to the number of its successors
        self.assign_cost(cost_settings)

        # set status values
        nx.set_node_attributes(
            self.g,
            values=0,
            name="status"
        )
    
    def assign_cost(self, cost_settings : dict):
        """Function to assign the social network graph node cost,
        given a defined policy: random proportional, proportional

        Args:
            cost_settings (dict): dictionary with cost settings parameters

        Raises:
            ValueError: invalid cost_name
        """
        # initialize attribute
        nx.set_node_attributes(self.g, values = 1, name = "cost")
        # set maximum degree
        max_degree = max(self.g.degree)[1]
        if cost_settings["cost_name"] == 'random_proportional':
            for node in self.g.nodes:
                n_successors = len(list(self.g.successors(node)))
                self.g.nodes[node]['cost'] = np.random.uniform(
                    low=0, high= n_successors / max_degree
                )
        elif cost_settings["cost_name"] == "proportional":
            for node in self.g.nodes:
                n_successors = len(list(self.g.successors(node)))
                self.g.nodes[node]['cost'] = (n_successors + 0.01) / max_degree * cost_settings["max_price"]
        else:
            raise ValueError("invalid cost_name")

    def get_n_influenced(self):
        """Get method for the influenced nodes number

        Returns:
            int: number of influenced nodes
        """
        return len(self.state)

    def get_cost(self, action : list):
        """Function to compute the total paid nodes cost

        Args:
            action (list): graph node set of the chosen nodes from the agent with get_action() function

        Returns:
            float: total paid nodes cost
        """
        tot = 0.0
        for node in action:
            tot += self.g.nodes[node]['cost']
        return tot

    def set_state(self, state: np.array):
        """Function to set the attribute status of the nodes to 1, meaning influenced nodes

        Args:
            state (np.array): graph nodes set of the influenced nodes
        """
        for node in self.g.nodes:
            if node in state:
                self.g.nodes[node]['status'] = 1

    def plot(self):
        """Plotting the social network graph, coloring influenced nodes
        """
        # define the colors:
        colors = []
        for _, state in nx.get_node_attributes(self.g,'status').items():
            colors.append('red' if state == 1 else 'blue')
        # draw the graph:
        nx.draw(
            self.g,
            with_labels=False,
            node_size=25,
            node_color=colors
        )
        plt.show()
