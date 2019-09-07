import os
from sys import exit, exc_info, argv
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#pip3 install git+https://github.com/slremy/netsapi --user --upgrade


from netsapi.challenge import *
# For a given environment, evaluate a policy by applying its evaluateReward method
#!pip install gaft
from gaft import GAEngine
#from gaft.components import BinaryIndividual, Population, DecimalIndividual, OrderIndividual
from gaft.components import Population, OrderIndividual

from gaft.operators import RouletteWheelSelection, UniformCrossover, FlipBitMutation, ExponentialRankingSelection, TournamentSelection, LinearRankingSelection
#Analysis plugin base class.
from gaft.plugin_interfaces.analysis import OnTheFlyAnalysis

import math
from random import random, uniform

if __name__ == '__main__':

    import random
    envSeqDec = ChallengeProveEnvironment(experimentCount = 20000) # Initialise a New Challenge Environment to post entire policy
    #env = ChallengeEnvironment(experimentCount = 20000)
    eps = 0.1 # equal to actions space resolution # range/eps
    pop_size = 2
    cross_prob = 0.6
    exchange_prob = 0.7
    mutation_pob = 1
    generation = 5
    REWARDS = []
    NEW = []
    POLICY = []
    tmp_reward = []
    tmp_policy = []
    policy_450 = []
    reward_generation = []
    bad_p = [{'1': [0.9, 0.3], '2': [0.6, 0.0], '3': [0.4, 0.0], '4': [0.8, 0.8], '5': [0.2, 0.6]}]
    good_p = []

    # random.seed(54)
    # best_action = ([0], [0.8, 1])
    
    turb = 0
    test_action = ([[i/10 for i in range(0, 11)],[i/10 for i in range(0,11)]])
    # test_action = ([0, 0.1, 0.2, 0.3, 0.4, 0.5], [0.6, 0.7, 0.8, 0.9, 1])
    
    # 2. Define population
    indv_template = OrderIndividual(ranges=[(0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1)], eps=eps, actions = test_action) # low_bit and high_bit
    population = Population(indv_template=indv_template, size = pop_size)
    population.init()  # Initialize population with individuals.

    # 3. Create genetic operators

    # Use built-in operators here.
    #selection = RouletteWheelSelection()
    selection = LinearRankingSelection()

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
        if reward >= 550 and policy not in policy_450:
            policy_450.append(policy)
        if reward >= 150 and policy not in good_p:
            good_p.append(policy)
        if reward <= 100 and policy not in bad_p:
            bad_p.append(policy)
        tmp_reward.append(reward)
        reward_generation.append(reward)
        tmp_policy.append(policy)
        print('Policy : ', policy)
        print(policy_450,'**************************good solution***************')
        #print(policy_bad,'**************************bad solution***************')


        return reward + uniform(-turb, turb)

    @engine.analysis_register
    class ConsoleOutput(OnTheFlyAnalysis):
        master_only = True
        interval = 1
        def register_step(self, g, population, engine):
            
            best_indv = population.best_indv(engine.fitness)
            msg = 'Generation: {}, best fitness: {:.3f}'.format(g + 1, engine.fmax)
            #best_reward = max(tmp_reward[g + pop_size * (generation - 1): g + pop_size * generation])
            #print(pop_size * (g - 0), pop_size * (g + 1),'&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
            REWARDS.append(max(reward_generation[pop_size * (g - 0): pop_size * (g + 1)]))
            #best_policy = POLICY[tmp_reward.index(best_reward)]
            #POLICY.append(best_policy)
            engine.logger.info(msg)
    
    engine.run(ng = generation, good_p = good_p, bad_p = bad_p)
    print(policy_450)
    x = list(range(len(tmp_reward)))
    plt.plot(x, tmp_reward)
    plt.title(f'Sequential Rewards')
    #plt.savefig(f'./res_geneticAlgorithm/Sequential_Rewards_eps:{eps}_popsize:{pop_size}_generation:{generation}_mutation_pob:{mutation_pob}_exchange_prob:{exchange_prob}_cross_prob:{cross_prob}.jpg')
    plt.show()
