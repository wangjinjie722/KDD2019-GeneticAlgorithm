from netsapi.challenge import *
import pandas as pd
import matplotlib.pyplot as plt
import time
import tensorflow as tf
import math

#seed = np.random.randint(50,60)
# np.random.seed(1)
# tf.set_random_seed(1)

#####################  hyper parameters  ####################
LR_A = 0.001# learning rate for actor
LR_C = 0.002# learning rate for critic
GAMMA = 0.99     # reward discount
TAU = 0.001    # soft replacement
MEMORY_CAPACITY = 100
BATCH_SIZE =16
###############################  DDPG  ####################################
class DDPG(object):
    def __init__(self, a_dim, s_dim, a_bound,):
        print('QWERTYUIOP{SDFGHJKL:"XCVBNM<>?SDFGHJKL:ERTYUIOP#$%^&*()_CVBNM<>DFGHJKL')
        tf.reset_default_graph()#reset network
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0
        self.sess = tf.Session()

        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound,
        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')

        self.a = self._build_a(self.S,)
        q = self._build_c(self.S, self.a, )
        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Actor')
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Critic')
        ema = tf.train.ExponentialMovingAverage(decay=1 - TAU)          # soft replacement

        def ema_getter(getter, name, *args, **kwargs):
            return ema.average(getter(name, *args, **kwargs))

        target_update = [ema.apply(a_params), ema.apply(c_params)]      #update parameter
        # soft update operation
        print(';;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;',self.pointer)
        a_ = self._build_a(self.S_, reuse=True, custom_getter=ema_getter)   # replaced target parameters
        # if self.pointer%5==0:
        q_ = self._build_c(self.S_, a_, reuse=True, custom_getter=ema_getter)

        a_loss = - tf.reduce_mean(q)  # maximize the q
        self.atrain = tf.train.AdamOptimizer(LR_A).minimize(a_loss, var_list=a_params)

        with tf.control_dependencies(target_update):    # soft replacement happened at here
            q_target = self.R + GAMMA * q_
            td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
            self.ctrain = tf.train.AdamOptimizer(LR_C).minimize(td_error, var_list=c_params)
        with tf.control_dependencies(target_update):    # soft replacement happened at here
            q_target_end = self.R
            td_error_end = tf.losses.mean_squared_error(labels=q_target_end, predictions=q)
            self.ctrain_end = tf.train.AdamOptimizer(LR_C).minimize(td_error_end, var_list=c_params)

        self.sess.run(tf.global_variables_initializer())

    def OU(self, x, mu=0, theta=0.15, sigma=0.2):
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

    def choose_action(self, s):
        return self.sess.run(self.a, {self.S: s[np.newaxis, :]})[0]

    def learn(self):
        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        bt = self.memory[indices, :]
        # print(bt)
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]

        self.sess.run(self.atrain, {self.S: bs})
        self.sess.run(self.ctrain, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_})
    def learn_end(self):
        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        bt = self.memory[indices, :]
        # print(bt)
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]

        self.sess.run(self.atrain, {self.S: bs})
        self.sess.run(self.ctrain_end, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_})

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

    def _build_a(self, s, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope('Actor', reuse=reuse, custom_getter=custom_getter):
            net = tf.layers.dense(s, 100, activation=tf.nn.relu, name='l1', trainable=trainable)
            a = tf.layers.dense(net, self.a_dim, activation=tf.nn.tanh, name='a', trainable=trainable)
            return tf.multiply(a, self.a_bound, name='scaled_a')

    def _build_c(self, s, a, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope('Critic', reuse=reuse, custom_getter=custom_getter):
            n_l1 = 100#30
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            return tf.layers.dense(net, 1, trainable = trainable)  # Q(s,a)
        
class CustomAgent:
    def __init__(self, environment):
        self.environment = environment
        self.action_one = None
        self.action_one_reward = -np.inf

    def generate(self):
        candidates = []
        rewards = []
        s_dim = 5
        state_dim = 5
        a_dim = 2
        a_bound = 1
        var = 1.3
        max_r = 0
        ddpg = DDPG(a_dim, s_dim, a_bound)
        t1 = time.time()
        try:
            # Agents should make use of 20 episodes in each training run, if making sequential decisions
            for i in np.arange(6) / (6 - 1):
                for j in np.arange(6) / (6 - 1):
                    self.environment.reset()
                    s, r, done, info = self.environment.evaluateAction([i, j])
                    if r > self.action_one_reward:
                        print('new action:', [i, j], 'reward:', r)
                        self.action_one_reward = r
                        self.action_one = [i, j]

            for i in range(12):
                self.environment.reset()
                ep_reward = 0
                temp_policys = {}
                state = 1
                s = np.zeros(state_dim)
                s[state - 1] = 1
                print("state %d\n" % state, s)
                for j in range(5):
                    if j==0:
                        a = np.array(self.action_one)
                    else:
                        a = ddpg.choose_action(s)
                        a = np.clip(np.random.normal(a, var), 0, 1)
                        # while a[0]+a[1]>1.75:
                        #     print("change action")
                        #     print("original action: ", a)
                        #     a = np.clip(np.random.normal(a, var), 0, 1)
                        #     print("new action: ", a)
                    temp_policys[str(state)] = a.tolist()
                    print(a)
                    s_, r, done, info = self.environment.evaluateAction(a.tolist())
                    if math.isnan(r):
                        # r = 0
                        # self.environment.rewards[-1] = 0
                        break
                    print("reward: ", r)
                    ep_reward += r
                    state = s_
                    print("state %d" % state)
                    s_ = np.zeros(state_dim)
                    if state < 6:
                        s_[state - 1] = 1
                        if r > 0:
                            for i in range(10):
                                ddpg.store_transition(s, a, r / 10, s_)
                        else:
                            ddpg.store_transition(s, a, r / 10, s_)
                        if r > max_r:
                            flag = 0
                            while flag < 20:
                                ddpg.store_transition(s, a, r / 10, s_)
                                flag += 1
                            max_r = r

                    s = s_
                    print(s)
                    if ddpg.pointer > 0:  # MEMORY_CAPACITY:
                        var *= .9995  # decay the action randomness
                        ddpg.learn()
                    if j == 4:
                        var *= .9995  # decay the action randomness
                        print('Episode:', i, ' Reward: %i' % int(ep_reward), 'Explore: %.2f' % var, )
                        break

                candidates.append(temp_policys)
                rewards.append(ep_reward)

            best_policy = candidates[np.argmax(rewards)]
            best_reward = np.max(rewards)
            policy_reward = self.environment.evaluatePolicy(best_policy)
            print("policy_reward: ", policy_reward)
            print('Running time: ', time.time() - t1, "秒")
            print('Best policy:', best_policy)
            print('Best reward:', np.max(rewards))
            # print("seed: ", seed)
            x = list(range(len(rewards)))
            plt.plot(x, rewards)
            plt.show()

            index = np.argmax(rewards)
            return candidates[index], rewards[index]

        except (KeyboardInterrupt, SystemExit):
            print(exc_info())

if __name__=='__main__':
    EvaluateChallengeSubmission(ChallengeProveEnvironment, CustomAgent, "my_submission.csv")
#     ChallengeSeqDecEnvironment
# ChallengeProveEnvironment

