import multiprocessing as mp
import logging
import aaac
import tensorflow as tf
import configuration as C


class Trainer(object):
    def __init__(self):
        self.sess = tf.Session(
            config=tf.ConfigProto(log_device_placement=False,
                                  allow_soft_placement=True))
        init = tf.initialize_all_variables()
        self.sess.run(init)
        self.create_workers()

    def create_workers(self):
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
            p = C.GAME(in_q, out_q, i)
            if i == 0:  # Reference to first game process
                self.game = p
                print(p.env)
            processes.append(p)

        for i in range(C.NUM_WORKERS):
            a3c = aaac.A3C(self.sess, i, self.game)
            a3cs.append(a3c)

        for p in processes:
            p.start()

        for i in range(C.MAX_FRAMES):
            # Get next incoming observation and reward
            msg = out_q.get()
            (worker_id, observation, reward, done) = msg
            print("Trainer received %s, %d from %s" % (observation, reward, worker_names[worker_id]))

            a3c = a3cs[worker_id]

            if a3c.is_training_step(done):
                a3c.process_training_step(reward, done)

            if done == False:
                action = a3c.process_observation(observation, reward, done)
                in_qs[worker_id].put(action)
            else:
                in_qs[worker_id].put('reset')

        for i in range(C.NUM_WORKERS):
            in_qs[i].put('stop')


if __name__ == '__main__':
    # mp.log_to_stderr(logging.DEBUG)
    trainer = Trainer()
