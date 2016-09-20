import time

import numpy as np

import configuration as C


# The Asynchronous Advantage Actor Critic Algorithm
class A3C(object):
    def __init__(self, sess, id, game, model):
        self.sess = sess
        self.game = game
        self.id = id
        self.total_steps = 0  # Total steps taken in this A3C instance
        self.episode_reward = 0.0
        self.action_size = game.env.action_space.n
        self.model = model
        self.reset()
        self.start_time = time.time()
        self.last_episode = 0

    def reset(self):
        self.observations = []
        self.actions = []
        self.values = []
        self.rewards = []
        self.step = 0

    def is_training_step(self, done):
        return self.step == C.MAX_ROLL_OUT

    def process_observation(self, observation, reward, done):
        pi, v = self.model.calc_pi_and_v(observation)
        # print("pi: %s" % pi)
        action = np.random.choice(len(pi), p=pi)

        if self.step > 0:  # Store reward from last action, not this one
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

        episode_reward_sofar = self.episode_reward
        if done == True:
            R = 0.0
            self.episode_reward = 0.0
        else:
            R = self.model.calc_v(self.observations[-1])[0]  # Bootstrap with value of last observation

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

        self.model.train(batch_observations, batch_actions, batch_advantages, batch_R)

        self.reset()

        return episode_reward_sofar
