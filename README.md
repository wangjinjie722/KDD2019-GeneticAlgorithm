# KDD2019-GeneticAlgorithm-LSTM

Let me introduce the rules. Or maybe you can find it here : https://compete.hexagon-ml.com/practice/rl_competition/37/#description

This is a game you should provide a five-year policy to control the malaria. For every year, you should provide a number that represents the number of mosquito curtain and another number that represents the number of pesticide. You are restricted to interactive with the environment up to 100 times. Design an algorithm and get the best policy that has the biggest reward.


### Here we provide a kind of genetic algorithm for Reinforcement Learning Track in KDD2019. Also some basic algorithm solution like Q-learning, SARSA, DQN, DDPG. Before u start reading our code, 

if ur not familiar with these algorithm, I highly recommend u to read the tutorials of Morvan Zhou.(https://morvanzhou.github.io/tutorials/machine-learning/reinforcement-learning/)



## Genetic Algorithm

### **The code is developed upon the base of gaft(https://github.com/PytLab/gaft).**

Normally, the search efficiency of GA is not as excepted. Below is the method we use to raise it. In this project, we used 36 times attempts to get the correct first year policy and then use GA to search the space. For the child generation, we use a LSTM to predict the reward of child to accelerate the convergence. And set up the step length by 0.2.

The champion solution in this competition also uses GA. Congratulations to him!
The keypoint for his solution is setting up the mutation policy like this: 

- father : x
- children : 1-x

This is really interesting and can be explained by there's a prior knowledge that the same policy in two years should be high, low, high, like that.
