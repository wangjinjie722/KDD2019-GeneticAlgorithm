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
    
    envSeqDec = ChallengeProveEnvironment(experimentCount = 30000) # Initialise a New Challenge Environment to post entire policy
    #env = ChallengeEnvironment(experimentCount = 10000)
    eps = 0.1 # equal to actions space resolution, eps is step size
    pop_size = 10
    cross_prob = 0.7
    exchange_prob = 0.7
    mutation_pob = 0.5
    generation = 300
    REWARDS = []
    NEW = []
    POLICY = []
    tmp_reward = []
    tmp_policy = []
    policy_450 = []
    #random.seed(54)
    turb = 0

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
        print(policy)
        reward = envSeqDec.evaluatePolicy(policy) # Action in Year 1 only
        if reward >= 500 and policy not in policy_450:
            policy_450.append(policy)
        print('Sequential Result : ', reward)
        tmp_reward.append(reward)
        tmp_policy.append(policy)
        tmp_single = []
        print(policy_450,'**************************good solution***************')
        return reward + uniform(-turb, turb)

    engine.run(ng = generation)
    print(policy_450)
    x = list(range(1,len(tmp_reward) + 1))
    plt.plot(x, tmp_reward)
    plt.title('Sequential Rewards by Genetic Algorithm')
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    #plt.savefig('./res_geneticAlgorithm/Sequential_Rewards_eps:' + str(eps) +'_popsize:' + str(pop_size) +'_generation:' + str(generation) +'.jpg')
    plt.show()
