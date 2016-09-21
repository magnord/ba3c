import multiprocessing as mp

import cv2
import gym
import numpy as np


class GymGame(mp.Process):
    def __init__(self, in_q, out_q, id):
        super().__init__()
        self.in_q = in_q
        self.out_q = out_q
        self.id = id
        self.name = "Worker-" + str(id)  # mp.current_process().name
        # print("%s init" % self.name)
        self.env = None  # set by subclass

    def process_observation(self, obs):
        raise NotImplementedError

    def act(self, action):
        # print("%s acting" % self.name)
        observation, reward, done, _info = self.env.step(action)
        # print('Step reward: %s, done %s' % (reward, done))
        # self.env.render()
        return self.process_observation(observation), reward, done

    def reset(self):
        observation = self.env.reset()
        return self.process_observation(observation)

    def run(self):
        obs = self.reset()
        reward = 0.0
        done = False
        # print("%s sends obs: %s, r: %f, done: %s" % (self.id, obs, reward, done))
        self.out_q.put([self.id, obs, reward, done])
        while (True):
            # print("%s waiting for action" % self.name)
            msg = self.in_q.get()
            # print("%s received %s" % (self.name, msg))
            if msg == 'reset':
                obs = self.reset()
                reward = 0.0
                done = False
            elif msg == 'stop':
                self.in_q.task_done()
                break
            else:
                obs, reward, done = self.act(msg)
            # print("%s sends obs: %s, r: %f, done: %s" % (self.id, obs, reward, done))
            self.in_q.task_done()
            self.out_q.put((self.id, obs, reward, done))


class CartPole(GymGame):
    def __init__(self, in_q, out_q, id, game_name):
        super().__init__(in_q, out_q, id)
        self.env = gym.make(game_name)

    def process_observation(self, obs):
        return obs


class Atari(GymGame):
    def __init__(self, in_q, out_q, id, game_name):
        super().__init__(in_q, out_q, id)
        self.env = gym.make(game_name)
        print(self.env.observation_space)
        self.state = []

    # Atari specific reset, init four stacked frames
    def reset(self):
        obs = self.env.reset()
        obs = self.process_image(obs)
        self.state = np.reshape(np.stack((obs, obs, obs, obs), axis=2), (84, 84, 4))
        return self.state

    def process_observation(self, obs):
        obs = self.process_image(obs)
        self.state = np.append(self.state[:, :, 1:], obs, axis=2)
        return self.state

    def process_image(self, obs):
        obs = self.img_downsample_and_crop(obs)
        obs = self.img_normalize(obs)
        return obs

    def img_downsample_and_crop(self, img_in):
        img_in = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img_in, dsize=(84, 110))
        assert img.shape == (110, 84)
        img = np.reshape(img, (110, 84, 1))
        assert img.shape == (110, 84, 1)
        # img_out = img[18:102, :]
        img_out = img[18:102, :, :]
        assert img_out.dtype == np.uint8
        assert img_out.shape == (84, 84, 1)
        return img_out

    def img_normalize(self, img_in):
        img_out = (1.0 / 255.0) * img_in.astype(np.float32)
        return img_out
