import multiprocessing as mp
import gym


class GymGame(mp.Process):
    def __init__(self, in_q, out_q, id):
        super().__init__()
        self.in_q = in_q
        self.out_q = out_q
        self.id = id
        self.name = "Worker-" + str(id)  # mp.current_process().name
        print("%s init" % self.name)

    def act(self, action):
        print("%s acting" % self.name)
        observation, reward, done, _info = self.env.step(self.env.action_space.sample())
        self.env.render()
        return observation, reward, done

    def reset(self):
        observation, reward, done, _info = self.env.reset()
        return observation, reward, done

    def run(self):
        obs, reward, done = self.reset()
        self.out_q.put((self.id, obs, reward, done))
        while (True):
            msg = self.in_q.get()
            print("%s received %s" % (self.name, msg))
            if msg == 'reset':
                obs, reward, done = self.reset()
            elif msg == 'stop':
                self.in_q.task_done()
                break
            else:
                obs, reward, done = self.act(msg)
            self.in_q.task_done()
            self.out_q.put((self.id, obs, reward, done))


class CartPole(GymGame):
    def __init__(self, in_q, out_q, id):
        self.env = gym.make('CartPole-v0')
        super().__init__(in_q, out_q, id)
