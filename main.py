#!/usr/bin/python3
# -*- coding: utf-8 -*-
import json
import numpy as np
from agents.dummyAgent import DummyAgent
from envs.dynamicSocialNetwork import DynamicSocialNetwork


if __name__ == '__main__':
    # load settings
    fp = open("./cfg/settings_barabasi_ICM.json", 'r')
    settings = json.load(fp)
    fp.close()

    # set numpy seed
    np.random.seed(settings["seed"])

    # create environment
    env = DynamicSocialNetwork(settings)

    # create agent
    dummy_agent = DummyAgent(env, settings)

    # initialize env
    done = False
    state = env.reset()

    # run simulation
    while not done:
        action = dummy_agent.get_action(state)
        print(f"action: {action}")
        state, reward, done, info = env.step(action)
        print("---")
        print(f"{info['current_time']}]state: {state}")
        print(f"reward: {reward}")
        print(f'budget: {env.budget}')
        print('\n')
    env.close()
