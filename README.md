# Math-based Reinforcement Learning for the Adaptive Budgeted Influence Maximization Problem

This library simulates the environment of the social influence problem. You can download all the libraries needed to run the project via the *requirements* file.

The project is organized into several folders:
- *agents*: It contains the general abstract class for the agent and *DummyAgent* as an implementation example.
- *cfg*: It contains a folder with all the configuration files that describe the instance characteristics.
- *envs*: It contains the class of the environment describing the network containing the users that we want to influence.
- *models*: It contains the class for the graph management.
- *results*: It contains the resulting file.
- *benchmarks*: It contains the raw results obtained for the paper.

Finally, in the main.py there is an example of how to launch the program.

## Citing Us
The paper describing the solution methods and the results is:

```Bibtex
@misc{,
  doi = {},
  url = {},
  author = {Fadda, Edoardo and Di Corso, Evelina and Brusco, Davide and Aelenei, Vlad Stefan and Balan Rares, Alexandru},
  keywords = {Optimization and Control (math.OC), FOS: Mathematics},
  title = {Math-based Reinforcement Learning for the Adaptive Budgeted Influence Maximization Problem},
  publisher = {},
  journal={},
  year = {2024},
  copyright = {}
}
```

## Code Structure

```bash
|____agents
| |____agent.py
| |______init__.py
| |____dummyAgent.py


|____envs
| |______init__.py
| |____influenceModel.py


|____models
| |______init__.py
| |____CTModel.py
| |____ITModel.py
| |____LTModel.py
| |____Sampler.py
| |____socialNetwork.py

|____diffusion_models
| |______init__.py
| |____CTModel.py
| |____ITModel.py
| |____LTModel.py

|____graph_models
| |______init__.py
| |____socialNetwork.py

```

## Instance generation

The instance is generated using as input data the json file in the folder *cfg*.
An example of a configuration file is the following:
~~~ json
{
    "time_horizon": 10,
    "budget": 1,
    "lambda": 0.1,
    "graph_type": {
        "name": "barabasi_albert",
        "n_nodes": 100,
        "stub": 10
    },
    "cost_type": {
        "cost_name": "proportional",
        "max_price": 1
    },
    "influence_model":{
        "type": "ICM",
        "max_influence_prob": 0.1
    },
    "seed":0
}
~~~
With the following meaning:
- *time_horizon*: length of the time horizon.
- *budget*: the available amount of money.
- *lambda*: it weighs how much is good to influence a single node.
- *cost_type*: it describes how the cost of the nodes is computed. Refer to the paper for more details.

In the following, we better analyze the *graph_type* and *influence_model* entries of the json file. 

### Graph model

The following graph models are supported:
- scale_free_graph.
- erdos_renyi_graph.
- watts_strogatz_graph.
- random_regular_graph.
- barabasi_albert_graph.
- powerlaw_cluster_graph.
- random_lobster.
- real_graph

Each one of them can be created by the following lines to be added in the configuration file:

~~~ json
    "graph_type": {
        "name": "erdos_renyi",
        "p": 0.8
    },
~~~

~~~ json
    "graph_type": {
        "name": "power",
        "stubs": 3, 
        "prob": 0.8
    },
~~~

~~~ json
    "graph_type": {
        "name": "scale_free_graph",
        "alfa": 0.01, 
        "beta": 0.39,
        "gamma": 0.6, 
        "delta_in": 3, 
        "delta_out": 4
    },
~~~

~~~ json
    "graph_type": {
        "name": "random_lobster",
        "p1": 0.7,
        "p2": 0.4
    },
~~~

~~~ json
    "graph_type": {
        "name": "regular_random",
        "node_degree": 5
    },
~~~

~~~ json
    "graph_type": {
        "name": "watt_strogatz",
        "n_neigh": 10,
        "edge_prob": 0.8
    },
~~~

~~~ json
    "graph_type": {
        "name": "real_graph",
        "file_path": "./real_instances/p2p-Gnutella04.txt"
    },
~~~

In the folder *real_instances* we add the *p2p-Gnutella04*, *Slashdot0811*, *Slashdot0902*, *soc-Epinions1*, *twitter_combined*, and *Wiki-Vote* since the size of the network enables them to be loaded. The other graphs used in the paper can be downloaded from the following links:
- [Lazovec](https://snap.stanford.edu/data/web-Stanford.html)
- [soc-Epinions1](https://snap.stanford.edu/data/soc-Epinions1.html)
- [soc-Slashdot0811](https://snap.stanford.edu/data/soc-Slashdot0811.html)
- [soc-Slashdot0902](https://snap.stanford.edu/data/soc-Slashdot0902.html)
- [twitter_combined](https://snap.stanford.edu/data/ego-Twitter.html)
- [wiki-Vote](https://snap.stanford.edu/data/wiki-Vote.html)
- [Gnutella (04)](https://snap.stanford.edu/data/p2p-Gnutella04.html)

### Influence models
The code supports three influence models:
- Influence Cascade Model (ICM)
- Linear Threshold Model (LTM)
- Complex Threshold Model (CTM)

The data related to the type of diffusion model must be specified in the field of the cfg file in one of the following ways:

~~~json
    "influence_model":{
        "type": "ICM",
        "max_influence_prob": 0.1
    } 
~~~

~~~json
    "influence_model":{
        "type": "LTM",
        "max_influence_power": 0.1
    } 
~~~

~~~ json
    "influence_model":{
        "type": "CTM",
        "percentage": [0.1, 0.2, 0.35, 0.35],
        "a": [5, 4, 3, 2],
        "b": [1, 2, 3, 4],
        "decays": [0.8, 0.6, 0.5, 0.1],
        "lambda": [0.7, 0.5, 0.4, 0.1],
        "resistance_noises": [0.05, 0.05, 0.05, 0.05]
    }
~~~
 
It is possible to generate new diffusion models by creating a new class similar to *CTModel*, *ICModel*, *LTModel*. The only requirement is for them to have the *apply_influence* method as follows:
~~~ python
def apply_influence(self, action : list):
    pass
~~~
where action is a list of the nodes influenced.


## Agents & Envs

When dealing with a multistage environment, we consider a sequential approach based on the [Gym](https://www.gymlibrary.dev/) framework. We refer to the [Gym](https://www.gymlibrary.dev/) documentation for a deeper analysis of the observation/action/step sequentiality.
