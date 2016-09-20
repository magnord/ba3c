import multiprocessing as mp

import gym


class GymGame(mp.Process):
    def __init__(self, in_q, out_q, id):
        super().__init__()
        self.in_q = in_q
        self.out_q = out_q
        self.id = id
        self.name = "Worker-" + str(id)  # mp.current_process().name
        # print("%s init" % self.name)

    def act(self, action):
        # print("%s acting" % self.name)
        observation, reward, done, _info = self.env.step(action)
        # print('Step reward: %s, done %s' % (reward, done))
        # self.env.render()
        return observation, reward, done

    def reset(self):
        observation = self.env.reset()
        return observation

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
    def __init__(self, in_q, out_q, id):
        self.env = gym.make('CartPole-v0')
        super().__init__(in_q, out_q, id)
