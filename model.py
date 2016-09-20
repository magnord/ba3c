import tensorflow as tf
import tensorflow.contrib.slim as slim

import configuration as C


class Model(object):
    def __init__(self, sess, game):
        self.sess = sess
        self.game = game
        self.action_size = game.env.action_space.n

        # Defined in game specifc subclass
        self.observation = None
        self.pi = None
        self.v = None

        self.define_pi_and_v()
        self.define_loss()

        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.learning_rate = tf.train.exponential_decay(C.INITIAL_LEARNING_RATE,
                                                        global_step=self.global_step,
                                                        decay_steps=C.MAX_STEPS,
                                                        decay_rate=C.LEARNING_RATE_DECAY,
                                                        staircase=False)
        # Create an optimizer.
        self.opt = tf.train.RMSPropOptimizer(self.learning_rate, decay=0.9, momentum=0.0, epsilon=1e-10)
        # self.opt = tf.train.GradientDescentOptimizer(self.learning_rate)
        # self.grads_and_vars = self.opt.compute_gradients(self.total_loss,
        #                                                 tf.trainable_variables())
        # print("Trainable variable: %s" % [v.name for v in tf.trainable_variables()])
        # clipped_grads_and_vars = [(tf.clip_by_norm(gv[0], C.GRAD_CLIP_NORM), gv[1]) for gv in self.grads_and_vars]
        # self.opt.apply_gradients(clipped_grads_and_vars, global_step=self.global_step)
        self.opt_op = self.opt.minimize(self.total_loss, global_step=self.global_step,
                                        var_list=tf.trainable_variables())
        init = tf.initialize_all_variables()
        self.sess.run(init)

    def define_loss(self):
        self.action = tf.placeholder(tf.float32, [None, self.action_size], name='action')  # One-hot actions
        self.advantage = tf.placeholder(tf.float32, [None], name='advantage')
        log_pi = tf.log(self.pi)  # tf.clip_by_value(self.pi, 1e-20, 1.0))  # NaN protection
        entropy = -tf.reduce_sum(self.pi * log_pi, reduction_indices=1)
        pi_loss = - tf.reduce_sum(
            tf.reduce_sum(tf.mul(log_pi, self.action), reduction_indices=1) * self.advantage + entropy * C.ENTROPY_BETA)
        self.R = tf.placeholder(tf.float32, [None], name='R')
        v_loss = tf.nn.l2_loss(self.R - self.v)  # TODO: Stop gradient?
        self.total_loss = pi_loss + 0.5 * v_loss

    def calc_pi_and_v(self, observation):
        pi, v = self.sess.run([self.pi, self.v], feed_dict={self.observation: [observation]})
        # pi: (1,action_size), v: (1)
        return pi[0], v[0]

    def calc_v(self, observation):
        v = self.sess.run([self.v], feed_dict={self.observation: [observation]})
        return v[0]

    def train(self, batch_observations, batch_actions, batch_advantages, batch_R):
        # print("Batch:\n%s\n%s\n%s\n%s" % (batch_observations, batch_actions, batch_advantages, batch_R))
        self.sess.run([self.total_loss, self.opt_op],
                      feed_dict={
                          self.observation: batch_observations,
                          self.action: batch_actions,
                          self.advantage: batch_advantages,
                          self.R: batch_R})

        # self.sess.run(self.opt_op)

    def define_pi_and_v(self):  # Implemented by subclasses
        raise NotImplementedError()


class CartPole(Model):
    def __init__(self, sess, game):
        super().__init__(sess, game)

    # Game specific observation and perception model
    def define_pi_and_v(self):
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            activation_fn=tf.nn.relu,
                            weights_initializer=tf.contrib.layers.xavier_initializer()):  # tf.truncated_normal_initializer(stddev=0.1)):
            self.observation = tf.placeholder(tf.float32, shape=[None, 4], name='observation')
            x = slim.fully_connected(self.observation, 64, scope='fc_1')
            self.pi = tf.nn.softmax(slim.fully_connected(x, self.action_size, activation_fn=None, scope='pi'))
            v0 = slim.fully_connected(x, 1, activation_fn=None, scope='v')
            self.v = tf.reshape(v0, [-1])
