import multiprocessing as mp
import env
import time
import logging

NUM_WORKERS = 3
MAX_FRAMES = 10000
GAME = env.CartPole


def create_workers():
    # Establish communication queues
    in_qs = []
    worker_names = []
    processes = []

    out_q = mp.Queue()

    for i in range(NUM_WORKERS):
        in_q = mp.JoinableQueue()
        in_qs.append(in_q)
        name = "Worker-"  + str(i)
        worker_names.append(name)
        p = GAME(in_q, out_q, i)
        processes.append(p)
    for p in processes:
        p.start()

    for i in range(MAX_FRAMES):
        for w in range(NUM_WORKERS):
            msg = out_q.get()
            (w_id, observation, reward, done) = msg
            print("Trainer received %s, %d from %s" % (observation, reward, worker_names[w_id]))
            if done is True:
                action = 'reset'
            else:
                action = predict_action(observation)
            in_qs[w_id].put(action)

    for i in range(NUM_WORKERS):
        in_qs[i].put('stop')


if __name__ == '__main__':
    # mp.log_to_stderr(logging.DEBUG)
    create_workers()
