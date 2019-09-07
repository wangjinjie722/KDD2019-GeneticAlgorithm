from gaft import GAEngine
from gaft.components import BinaryIndividual, Population, DecimalIndividual
from gaft.operators import RouletteWheelSelection, UniformCrossover, FlipBitMutation, ExponentialRankingSelection
import math
from random import random, uniform
from gaft.plugin_interfaces.analysis import OnTheFlyAnalysis
from netsapi.challenge import *
import random
import matplotlib.pyplot as plt


from sys import exit, exc_info, argv
import numpy as np
import pandas as pd
#!pip install gaft --user --upgrade
#!pip3 install git+https://github.com/slremy/netsapi --user --upgrade

from netsapi.challenge import *

class GeneticAlgorithm:
    def __init__(self, environment):
        self.environment = environment
    
    def generate(self):
        
        best_policy = None
        best_reward = -float('Inf')
        candidates = []
        eps = 1 # equal to actions space resolution, eps is step size
        pop_size = 4
        cross_prob = 1
        exchange_prob = 1
        mutation_pob = 1
        generation = 4
        tmp_reward = []
        tmp_policy = []
        random.seed(54)
        turb = 5
        
        try:
            # Agents should make use of 20 episodes in each training run, if making sequential decisions

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
                policy = {'1': [p[0], p[1]], '2': [p[2], p[3]], '3': [p[4], p[5]], '4': [p[6], p[7]], '5': [p[8], p[9]]}xw
                reward = self.environment.evaluatePolicy(policy) # Action in Year 1 only
                print('Sequential Result : ', reward)
                tmp_reward.append(reward)
                tmp_policy.append(policy)
                tmp_single = []
                return reward + uniform(-turb, turb)
            
            # run
            engine.run(ng = generation)
            best_reward = max(tmp_reward)
            best_policy = tmp_policy[-pop_size]
        
        except (KeyboardInterrupt, SystemExit):
            print(exc_info())
        
        return best_policy, best_reward

if __name__ == "__main__":
    EvaluateChallengeSubmission(ChallengeSeqDecEnvironment, GeneticAlgorithm, "tutorial.csv")
    

