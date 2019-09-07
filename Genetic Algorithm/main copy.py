import os
from sys import exit, exc_info, argv
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#!pip install git+https://github.com/slremy/netsapi --user --upgrade

from netsapi.challenge import *
# For a given environment, evaluate a policy by applying its evaluateReward method
#!pip install gaft
from gaft import GAEngine
#from gaft.components import BinaryIndividual, Population, DecimalIndividual, OrderIndividual
from gaft.components import Population, OrderIndividual

from gaft.operators import RouletteWheelSelection, UniformCrossover, FlipBitMutation, ExponentialRankingSelection
#Analysis plugin base class.
from gaft.plugin_interfaces.analysis import OnTheFlyAnalysis

import math
from random import random, uniform

if __name__ == '__main__':

    import random
    envSeqDec = ChallengeProveEnvironment(experimentCount = 20000) # Initialise a New Challenge Environment to post entire policy
    env = ChallengeEnvironment(experimentCount = 20000)
    eps = 0.2 # equal to actions space resolution # range/eps
    pop_size = 4
    cross_prob = 1
    exchange_prob = 1
    mutation_pob = 1
    generation = 20
    REWARDS = []
    NEW = []
    POLICY = []
    tmp_reward = []
    tmp_policy = []

    #random.seed(54)

    turb = 0
    test_action = ([[i/100 for i in range(0, 21)],[i/100 for i in range(80,101)]])
    test_action = ([0, 0.1,0.2,0.3, 0.4, 0.5], [0.6,0.7, 0.8,0.9, 1])
    test_action = ([0.1], [0.5])
    #test_action = ([0, 0.2, 0.4, 0.6, 0.8, 1], [0, 0.2, 0.4, 0.6, 0.8, 1])
    #best_action = ([0], [0.8, 1])
    # 2. Define population
    indv_template = OrderIndividual(ranges=[(0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1)], eps=eps, actions = test_action) # low_bit and high_bit
    population = Population(indv_template=indv_template, size = pop_size)
    population.init()  # Initialize population with individuals.

    # 3. Create genetic operators

    # Use built-in operators here.
    selection = RouletteWheelSelection()
    crossover = UniformCrossover(pc=cross_prob, pe=exchange_prob) # PE = Gene exchange probability
    mutation = FlipBitMutation(pm=mutation_pob) # 0.1 todo The probability of mutation

    # 4. Create genetic algorithm engine to run optimization

    engine = GAEngine(population=population, selection=selection,
                    crossover=crossover, mutation=mutation,)
                    # analysis=[FitnessStore])
        
        
    # 5. Define and register fitness function   
    @engine.fitness_register
    #@engine.dynamic_linear_scaling(target='max', ksi0=2, r=0.9)

    def fitness(indv):
        p = [0 for _ in range(10)]
        p = indv.solution
        # encode
        policy = {'1': [p[0], p[1]], '2': [p[2], p[3]], '3': [p[4], p[5]], '4': [p[6], p[7]], '5': [p[8], p[9]]}
        reward = envSeqDec.evaluatePolicy(policy) # Action in Year 1 only
        print('Sequential Result : ', reward)
        tmp_reward.append(reward)
        tmp_policy.append(policy)
        print('Policy : ', policy)

        return reward + uniform(-turb, turb)

    @engine.analysis_register
    class ConsoleOutput(OnTheFlyAnalysis):
        master_only = True
        interval = 1
        def register_step(self, g, population, engine):
            best_indv = population.best_indv(engine.fitness)
            msg = 'Generation: {}, best fitness: {:.3f}'.format(g + 1, engine.fmax)
            best_reward = max(tmp_reward[g : g + pop_size * generation])
            REWARDS.append(best_reward)
            #best_policy = POLICY[tmp_reward.index(best_reward)]
            #POLICY.append(best_policy)
            engine.logger.info(msg)
        
    engine.run(ng = generation)
    x = list(range(len(tmp_reward)))
    plt.plot(x, tmp_reward)
    plt.title('Sequential Rewards by GA')
    #plt.savefig(f'./res_geneticAlgorithm/Sequential_Rewards_eps:{eps}_popsize:{pop_size}_generation:{generation}_mutation_pob:{mutation_pob}_exchange_prob:{exchange_prob}_cross_prob:{cross_prob}.jpg')
    plt.show()
