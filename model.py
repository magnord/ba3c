import configuration as C
import tensorflow as tf
import tensorflow.contrib.slim as slim


class Model(object):
    def __init__(self, game, sess):
        self.sess = sess
        self.game = game
        self.action_size = game.env.action_space.n

        # Defined in game specifc subclass
        self.observation = None
        self.pi = None
        self.v = None

        self.define_loss()
        self.define_pi_and_v()

        self.training_step = tf.Variable(0, name='training_step', trainable=False)
        learning_rate = tf.train.exponential_decay(C.INITIAL_LEARNING_RATE,
                                                   self.training_step,
                                                   C.MAX_STEPS,
                                                   C.LEARNING_RATE_DECAY,
                                                   staircase=True)
        # Create an optimizer.
        # opt = tf.train.RMSPropOptimizer.__init__(learning_rate, decay=0.9, momentum=0.0, epsilon=1e-10)
        self.opt = tf.train.GradientDescentOptimizer(learning_rate)

    def define_loss(self):
        self.define_loss()

        self.action = tf.placeholder("float", [None, self.action_size])  # One-hot actions
        self.advantage = tf.placeholder("float", [None])
        log_pi = tf.log(self.pi)  # tf.clip_by_value(self.pi, 1e-20, 1.0))  # NaN protection
        entropy = -tf.reduce_sum(self.pi * log_pi, reduction_indices=1)
        pi_loss = - tf.reduce_sum(
            tf.reduce_sum(tf.mul(log_pi, self.action), reduction_indices=1) * self.advantage + entropy * C.ENTRPOY_BETA)
        self.R = tf.placeholder("float", [None])
        v_loss = tf.nn.l2_loss(self.R - self.v)  # TODO: Stop gradient?
        self.total_loss = pi_loss + 0.5 * v_loss

    def calc_pi_and_v(self, observation):
        pi, v = self.sess.run([self.pi, self.v], feed_dict={self.observation: [observation]})
        # pi: (1,action_size), v: (1)
        return pi[0], v[0]

    def train(self, batch_observations, batch_actions, batch_advantages, batch_R):
        self.sess.run(self.total_loss,
                      feed_dict={
                          self.observation: batch_observations,
                          self.action: batch_actions,
                          self.advantage: batch_advantages,
                          self.R: batch_R})
        self.grads_and_vars = self.opt.compute_gradients(self.total_loss,
                                                         tf.trainable_variables())  # TODO: Check trainable vars is correct set
        print("Trainable variable: %s" % tf.trainable_variables())
        clipped_grads_and_vars = [(tf.clip_by_norm(gv[0], C.GRAD_CLIP_NORM), gv[1]) for gv in self.grads_and_vars]
        self.opt.apply_gradients(clipped_grads_and_vars)

    def define_pi_and_v(self):  # Implemeted by subclasses
        raise NotImplementedError()


class CartPole(Model):
    def __init__(self, game, sess):
        super().__init__(game, sess)

    # Game specific observation and perception model
    def define_pi_and_v(self):
        self.observation = tf.placeholder(shape=(None, 4), name='observation')
        x = slim.fully_connected(self.observation, 64, scope='fc_1')
        self.pi = slim.fully_connected(x, self.action_size, activation_fn=None)
        self.v = slim.fully_connected(x, 1, activation_fn=None)
