import multiprocessing as mp
import time

import tensorflow as tf

import aaac
import conf as C


class Trainer(object):
    def __init__(self):
        self.sess = tf.Session(
            config=tf.ConfigProto(log_device_placement=False,
                                  allow_soft_placement=True))
        self.create_and_start_workers()
        # these variables are set later
        self.game = None
        self.model = None

    def create_and_start_workers(self):
        # Establish communication queues
        in_qs = []
        worker_names = []
        processes = []
        a3cs = []

        out_q = mp.Queue()

        for i in range(C.NUM_WORKERS):
            in_q = mp.JoinableQueue()
            in_qs.append(in_q)
            name = "Worker-" + str(i)
            worker_names.append(name)
            p = C.GAME(in_q, out_q, i, C.GAME_NAME)
            if i == 0:  # Reference to first game process
                self.game = p
            processes.append(p)

        print(self.game)
        self.model = C.MODEL(self.sess, self.game)

        for i in range(C.NUM_WORKERS):
            a3c = aaac.A3C(self.sess, i, self.game, self.model)
            a3cs.append(a3c)

        for p in processes:
            p.start()

        action_step = last_action_step = 0
        start_time = time.time()
        training_step = 0
        for i in range(int(C.MAX_STEPS)):
            # Get next incoming observation and reward
            msg = out_q.get()
            [worker_id, observation, reward, done] = msg
            # print("Trainer received %s, %d from %s" % (observation, reward, worker_names[worker_id]))

            a3c = a3cs[worker_id]

            if a3c.is_training_step(done) or done == True:
                training_step += 1
                episode_reward = a3c.process_training_step(reward, done)

            if done == False:
                action = a3c.process_observation(observation, reward, done)
                action_step += 1
                # print('Trainer send action: %s to %s' % (action, worker_names[worker_id]))
                in_qs[worker_id].put(action)
            else:  # Episode terminated
                in_qs[worker_id].put('reset')

                # Statistics
                end_time = time.time()
                elapsed_time = end_time - start_time
                if elapsed_time > 10.0:
                    steps = action_step - last_action_step
                    last_action_step = action_step
                    start_time = end_time
                    print("%7.0d %6.0d Reward: %3d, lr: %0.6f, %5.0d step/s" % (action_step,
                                                                                training_step,
                                                                                episode_reward,
                                                                                self.sess.run(self.model.learning_rate),
                                                                                steps / elapsed_time))

        for i in range(C.NUM_WORKERS):
            in_qs[i].put('stop')


if __name__ == '__main__':
    # mp.log_to_stderr(logging.DEBUG)
    trainer = Trainer()
