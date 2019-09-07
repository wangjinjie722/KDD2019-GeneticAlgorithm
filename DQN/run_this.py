# from maze_env import Maze
# from RL_brain import DeepQNetwork
from DQN_modified import DeepQNetwork
from netsapi.challenge import *
import matplotlib.pyplot as plt

def run_maze():
    step = 0
    rewards=[]
    for episode in range(5):#5
        try:
            actionList = [[0, 0]]
            # initial observation
            envSeqDec.reset()
            observation = np.array([1])
            observation=stateOnehot(observation[0])
            # observation.extend(actionList[-1])
            observation=np.array(observation)
            reward_step = 0
            for j in range(5):
                # fresh env
                # env.render()

                # RL choose action based on observation
                action = RL.choose_action(observation)

                # RL take action and get next observation and reward
                # print(action)
                a=action_space[action]
                a=np.array(a)
                # a=a+OU(a)/10   #添加噪声
                a=np.clip(a,0,1)
                print(a)
                # if j==4:
                #     a=np.array([0.0,1.0])
                print('///////////',j,']]]]',a)
                observation_, reward, done,info = envSeqDec.evaluateAction(a)
                actionList.append(a)
                print(observation_, reward, done,info)
                observation_=stateOnehot(observation_)#状态onehot
                # observation_.extend(actionList[-1])#state扩充上一个action
                observation_=np.array(observation_)
                reward_step+=reward
                # if reward>80:
                RL.store_transition(observation, action, reward, observation_)
                print(observation, action, reward, observation_)

                if (step > 20) and (step % 5 == 0):
                    RL.learn()

                # swap observation
                observation = observation_

                # break while loop when end of this episode
                if done:
                    print('Episode:', episode, ' Reward: %i' % int(reward_step) )
                    # RL.learn()
                    break
                step += 1
            rewards.append(reward_step)
        except Exception:
            return rewards
    return rewards
    # end of game
# print('game over')
    # env.destroy()

def stateOnehot(s):
    if s==1:
        res=[1,0,0,0,0,0]
    elif s==2:
        res=[0,1,0,0,0,0]
    elif s==3:
        res=[0,0,1,0,0,0]
    elif s==4:
        res=[0,0,0,1,0,0]
    elif s==5:
        res=[0,0,0,0,1,0]
    elif s==6:
        res=[0,0,0,0,0,1]
    return res
'''
生成action space
'''
def action2Index():
    action1 = list(range(0, 110, 20))
    action1 = [each / 100 for each in action1]
    actionListValue = []
    for i in action1:
        for j in action1:
            actionListValue.append(([i, j]))
    # print(len(actionList))
    # actionList = list(range(len(actionListValue)))
    return actionListValue
def getActionSpace(envSeqDec):
    oriSpace = action2Index()
    print(len(oriSpace))
    # envSeqDec = ChallengeSeqDecEnvironment(experimentCount=20000)
    actionSpace = []
    res = {}
    envSeqDec.state = 1
    for each in oriSpace:
        a = np.array(each)
        observation_, reward, done, info = envSeqDec.evaluateAction(a)
        print(observation_, reward, done, info)
        # print('--------------',envSeqDec.action)
        envSeqDec.reset()
        if reward!=reward:
            continue
        res[tuple(a)] = reward
    print('===========', sorted(res.items(), key=lambda x: x[1], reverse=True))
    action1 = sorted(res.items(), key=lambda x: x[1], reverse=True)[:1]
    action1=list(action1[0][0])
    print('action1:', action1)
    actionSpace.append(action1)
    # out.append(action1)
    res2 = {}
    for each in oriSpace:
        a = np.array(each)
        envSeqDec.state = 2
        envSeqDec.action = action1#list(action1[0][0])
        # print('///////////////',envSeqDec.action)
        observation_, reward, done, info = envSeqDec.evaluateAction(a)
        print(observation_, reward, done, info)
        envSeqDec.state = 2
        envSeqDec.action = action1#list(action1[0][0])
        if reward!=reward:
            continue
        res2[tuple(a)] = reward
    print('===========', sorted(res2.items(), key=lambda x: x[1], reverse=True))
    action2 = sorted(res2.items(), key=lambda x: x[1], reverse=True)[:1]
    print('action2:', action2)
    action2=list(action2[0][0])
    actionSpace.append(action2)
    print('actionSpace:---------------',actionSpace)
    return actionSpace
'''
OU噪声
'''
def OU(x, mu=0, theta=0.15, sigma=0.2):
    """Ornstein-Uhlenbeck process.
    formula：ou = θ * (μ - x) + σ * w

    Arguments:
        x: action value.
        mu: μ, mean fo values.
        theta: θ, rate the variable reverts towards to the mean.
        sigma：σ, degree of volatility of the process.

    Returns:
        OU value
    """
    return theta * (mu - x) + sigma * np.random.randn(1)

if __name__ == "__main__":
    # maze game
    # env = Maze()
    # action_space=[[0.0,1.0], [1.0,0.0], [1.0,1.0],[0,0], [0,0.8],[0.8,0]]
    # action_space=action2Index()
    # action_space=[[0.2,1.0],[0.8,0.0],[0.0,0.8],[0.0,1.0],[1.0,0.0]]
    # action_space=[[1.0,0.0],[0.0,1.0],[0.0,0.8]]
    # print(action_space)
    # action_space=[[0.0, 0.8], [1.0, 0.0]]
    # action_space = [[0.0, 1.0], [1.0, 0.8]]
    # action_space = [[1.0, 1.0], [1.0, 0.0],[0.0,0.0],[0.0,1.0]]
    # action_space=[[0.0,0.8],[0.0,1.0]]
    # print(action_space)
    # print(action_space[:4])
    # print(action_space[5:])
    # print(action_space[:-6])
    # print(action_space[-5:])
    envSeqDec = ChallengeProveEnvironment()
    action_space=getActionSpace(envSeqDec)#缩减action space
    print('actionSpace::::::::::::',action_space)
    RL = DeepQNetwork(len(action_space), 6,
                      learning_rate=0.0001,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=20,#200
                      memory_size=100,#2000
                      batch_size=16,
                      # output_graph=True
                      )
    # print(RL)
    # print('\n'.join(['%s:%s' % item for item in RL.__dict__.items()]))
    rewards=run_maze()
    print('Best Reward:',np.max(rewards))
    x = list(range(len(rewards)))
    plt.plot(x, rewards)
    plt.show()

    # env.mainloop()
    # RL.plot_cost()