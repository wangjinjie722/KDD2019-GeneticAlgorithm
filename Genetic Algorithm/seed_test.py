"""
Author: Kai Wang
Data: 06/25/2019
"""

from gaft import GAEngine
from gaft.components import BinaryIndividual, Population, DecimalIndividual
from gaft.operators import RouletteWheelSelection, UniformCrossover, FlipBitMutation, ExponentialRankingSelection
import math
from random import random, uniform
from gaft.plugin_interfaces.analysis import OnTheFlyAnalysis
from netsapi.challenge import *
import random
import matplotlib.pyplot as plt

if __name__ == "__main__":
    candidate = [49, 55, 103, 187, 199, 20, 30, 54, 76, 124, 180]
    res_11 = {}
    mid = {}
    for c in candidate:
        res_11[c] = []
    mid = res_11
    for c in candidate:
        for runs in range(10):
            print(f'******************{candidate.index(c)}*****{runs}*****')
            envSeqDec = ChallengeSeqDecEnvironment(experimentCount = 10000) # Initialise a New Challenge Environment to post entire policy
            env = ChallengeEnvironment(experimentCount = 10000)
            eps = 1 # equal to actions space resolution, eps is step size
            pop_size = 4
            cross_prob = 1
            exchange_prob = 1
            mutation_pob = 1
            generation = 4
            REWARDS = []
            NEW = []
            POLICY = []
            tmp_reward = []
            tmp_policy = []
            random.seed(c)
            turb = 5

            # Define population
            indv_template = DecimalIndividual(ranges=[(0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1),(0, 1), (0, 1)], eps=eps)
            population = Population(indv_template=indv_template, size = pop_size)
            population.init()  # Initialize population with individuals.

            # Create genetic operators
            # Use built-in operators here.
            selection = RouletteWheelSelection()
            crossover = UniformCrossover(pc=cross_prob, pe=exchange_prob) # PE = Gene exchange probability
            mutation = FlipBitMutation(pm=mutation_pob) # 0.1 todo The probability of mutation

            # Create genetic algorithm engine to run optimization
            engine = GAEngine(population=population, selection=selection,
                              crossover=crossover, mutation=mutation,)

            # Define and register fitness function
            @engine.fitness_register
            def fitness(indv):
                p = [0 for _ in range(10)]
                p = indv.solution
                policy = {'1': [p[0], p[1]], '2': [p[2], p[3]], '3': [p[4], p[5]], '4': [p[6], p[7]], '5': [p[8], p[9]]}
                reward = envSeqDec.evaluatePolicy(policy) # Action in Year 1 only
                print('Sequential Result : ', reward)
                tmp_reward.append(reward)
                tmp_policy.append(policy)
                tmp_single = []
                return reward + uniform(-turb, turb)

            engine.run(ng = generation)
            res_11[c].append(max(tmp_reward))
            if max(tmp_reward) < 440:
                break
        res_11[c].sort()
        if len(res_11[c]) > 9:
            mid[c] = (res_11[c][4] + res_11[c][5]) / 2
    print(f'**************************{mid}****')
