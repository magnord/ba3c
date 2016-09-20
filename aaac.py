import numpy as np
import tensorflow as tf
import configuration as C
import model


# The Asynchronous Advantage Actor Critic Algorithm
class A3C(object):
    def __init__(self, sess, id, game):
        self.sess = sess
        self.game = game
        self.id = id
        self.total_steps = 0  # Total steps taken in this A3C instance
        self.action_size = game.env.action_space.n
        self.reset()

    def reset(self):
        self.observations = []
        self.actions = []
        self.values = []
        self.rewards = []
        self.step = 0
        self.episode_reward = 0.0

    def is_training_step(self, done):
        return self.step == C.MAX_ROLL_OUT or done == True

    def process_observation(self, observation, reward, done):

        pi, v = model.calc_pi_and_v(self.sess, observation)
        action = np.random.choice(len(pi), p=pi)
        self.total_steps += 1

        if len(self.observations) > 0:  # Store reward from last action, not this one
            self.episode_reward += reward
            self.rewards.append(reward)  # np.clip(reward, -1, 1))

        self.observations.append(observation)
        self.actions.append(action)
        self.values.append(v)

        self.total_steps += 1
        self.step += 1

        # let game process action
        return action

    def process_training_step(self, reward, done):

        # Add reward for the last action before the training step
        self.episode_reward += reward
        self.rewards.append(reward)  # np.clip(reward, -1, 1))

        if done == True:
            R = 0.0
            # TODO: print self.episode_reward
        else:
            R = self.values[-1]  # Bootstrap with value of last observation

        self.actions.reverse()
        self.observations.reverse()
        self.rewards.reverse()
        self.values.reverse()

        batch_observations = []
        batch_actions = []
        batch_advantages = []
        batch_R = []

        for (ai, ri, si, vi) in zip(self.actions, self.rewards, self.observations, self.values):
            R = ri + C.GAMMA * R
            advantage = R - vi
            action_one_hot = np.zeros([self.action_size])
            action_one_hot[ai] = 1

            batch_observations.append(si)
            batch_actions.append(action_one_hot)
            batch_advantages.append(advantage)
            batch_R.append(R)

        model.train(batch_observations, batch_actions, batch_advantages, batch_R)

        self.reset()
