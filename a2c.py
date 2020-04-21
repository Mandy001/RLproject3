import numpy as np
import tensorflow as tf

from GridWorld import GridWorld

np.random.seed(20)
tf.set_random_seed(20)

MAX_EPISODE = 1000
MAX_EP_STEPS = 1000  # maximum time step in one episode
GAMMA = 0.9  # reward discount in TD error
lr_actor = 0.001
lr_critic = 0.01

grid_world_h = 5
grid_world_w = 5
env = GridWorld(grid_world_h, grid_world_w)

n_features = 2
n_actions = 4


class Actor(object):
    def __init__(self, sess, n_features, n_actions, lr=0.001):
        self.sess = sess
        self.state = tf.placeholder(tf.float32, [1, n_features], "state")
        self.action = tf.placeholder(tf.int32, None, "act")
        self.td_error = tf.placeholder(tf.float32, None, "td_error")

        with tf.variable_scope('Actor'):
            state_layer = tf.layers.dense(
                inputs=self.state,
                units=20,
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0., .1),
                bias_initializer=tf.constant_initializer(0.1),
                name='actor_state_layer'
            )

            self.acts_prob = tf.layers.dense(
                inputs=state_layer,
                units=n_actions,
                activation=tf.nn.softmax,
                kernel_initializer=tf.random_normal_initializer(0., .1),
                bias_initializer=tf.constant_initializer(0.1),
                name='action_prob'
            )

        with tf.variable_scope('exp_v'):
            log_prob = tf.log(self.acts_prob[0, self.action])
            self.exp_v = tf.reduce_mean(log_prob * self.td_error)

        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(-self.exp_v)

    def learn(self, state, action, td):
        state = state[np.newaxis, :]
        feed_dict = {self.state: state, self.action: action, self.td_error: td}
        _, exp_v = self.sess.run([self.train_op, self.exp_v], feed_dict)
        return exp_v

    def choose_action(self, state):
        state = state[np.newaxis, :]
        probs = self.sess.run(self.acts_prob, {self.state: state})
        return np.random.choice(np.arange(probs.shape[1]), p=probs.ravel())


class Critic(object):
    def __init__(self, sess, n_features, lr=0.01):
        self.sess = sess

        self.state = tf.placeholder(tf.float32, [1, n_features], "state")
        self.next_value = tf.placeholder(tf.float32, [1, 1], "next_value")
        self.reward = tf.placeholder(tf.float32, None, 'r')

        with tf.variable_scope('Critic'):
            state_layer = tf.layers.dense(
                inputs=self.state,
                units=20,
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0., .1),
                bias_initializer=tf.constant_initializer(0.1),
                name='critic_state_layer'
            )

            self.value = tf.layers.dense(
                inputs=state_layer,
                units=1,
                activation=None,
                kernel_initializer=tf.random_normal_initializer(0., .1),
                bias_initializer=tf.constant_initializer(0.1),
                name='q_value'
            )

        with tf.variable_scope('squared_TD_error'):
            self.td_error = self.reward + GAMMA * self.next_value - self.value
            self.loss = tf.square(self.td_error)
        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)

    def learn(self, state, reward, next_state):
        state, next_state = state[np.newaxis, :], next_state[np.newaxis, :]

        next_value = self.sess.run(self.value, {self.state: next_state})
        td_error, _ = self.sess.run([self.td_error, self.train_op],
                                    {self.state: state, self.next_value: next_value, self.reward: reward})
        return td_error


sess = tf.Session()

actor = Actor(sess, n_features=n_features, n_actions=n_actions, lr=lr_actor)
critic = Critic(sess, n_features=n_features, lr=lr_critic)

sess.run(tf.global_variables_initializer())

for i_episode in range(MAX_EPISODE):
    _, state = env.reset()
    step = 0
    track_r = []
    while True:

        action = actor.choose_action(state)
        _, next_state, reward, done = env.step(action)
        env.render()
        track_r.append(reward)

        td_error = critic.learn(state, reward, next_state)
        actor.learn(state, action, td_error)
        state = next_state
        step += 1

        if done or step >= MAX_EP_STEPS:
            ep_rs_sum = sum(track_r)
            if 'running_reward' not in globals():
                running_reward = ep_rs_sum
            else:
                running_reward = running_reward * 0.95 + ep_rs_sum * 0.05
            print("episode:", i_episode, "step:", step, "  reward:", int(running_reward))
            break

print('value table:')
for i in range(grid_world_w):
    for j in range(grid_world_h):
        state = np.array([[i, j]])
        q_value = sess.run(critic.value, {critic.state: state})
        print(np.squeeze(state), np.squeeze(q_value))
