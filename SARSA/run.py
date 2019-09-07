"""
Sarsa is a online updating method for Reinforcement learning.

Unlike Q learning which is a offline updating method, Sarsa is updating while in the current trajectory.

You will see the sarsa is more coward when punishment is close because it cares about all behaviours,
while q learning is more brave because it only cares about maximum behaviour.
"""

from RL_brain import SarsaTable
from netsapi.challenge import *
import matplotlib.pyplot as plt

random.seed(197)

def generate():
    
    action_space = [[0.0, 1.0], [1.0, 0.0], [0.0, 0.0], [1.0, 1.0]]
    #action_space = [[0.0, 1.0], [1.0, 0.0]]
    rewards_20 = []
    policy_20 = []
    rewards_seq = []
    
    for episode in range(20):
        
        # initial observation
        envSeqDec.reset()
        observation =1
        rewards=0
        policy={}
        
        for j in range(5):

            action = RL.choose_action(str(observation))
            a=action_space[action-1]
            print(';;;;;;;;;;;',j,']]]]',a)
            policy[str(j+1)]=a
            # RL take action and get next observation and reward
            observation_, reward, done, info = envSeqDec.evaluateAction(a)
            if reward:
                rewards+=reward
            if not reward:
                pass
            print(observation_, reward, done, info)
            action_ = RL.choose_action(str(observation_))
            
            # RL learn from this transition
            RL.learn(str(observation), action, reward, str(observation_), action_)
            
            # swap observation
            observation = observation_
            
            # break while loop when end of this episode
            if done:
                print('Episode:', episode + 1, ' Reward: %i' % int(rewards))
                print('Policy:', policy)
                break
                
        rewards_20.append(rewards)
        policy_20.append(policy)
        
    print('Best Reward:',np.max(rewards_20))
    print('Best Policy:',policy_20[np.argmax(rewards_20)])
    x = list(range(len(rewards_20)))
    plt.plot(x, rewards_20)
    #plt.title(f'Sarsa Result action_space: {action_space} learn_rate: {learning_rate} reward_decay: {reward_decay} e_greedy: {e_greedy}')
    plt.title('Sarsa Result')
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.show()


if __name__ == "__main__":
    envSeqDec = ChallengeSeqDecEnvironment(experimentCount=20000)
    action_space = [1,2,3,4] # Using two actions can get steady 490 - 500 result.
    RL = SarsaTable(actions=action_space,
                learning_rate=0.05,
                reward_decay=0.5,
                e_greedy=0.9)
    generate()

