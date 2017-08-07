import argparse
import ctypes
import time

import cv2  # hack: unused, but this must load before torch
import universe
from torch import multiprocessing
from torch import optim

from a3c import A3C
from env import create_atari_env
from model import Model
from visualizer import Visualizer

universe.configure_logging()

parser = argparse.ArgumentParser(description="Run commands")
parser.add_argument('-w', '--num-workers', default=1, type=int,
                    help="Number of workers")
parser.add_argument('-m', '--num-envs', default=1, type=int,
                    help="Number of environments to run on each worker")
parser.add_argument('-e', '--env-id', type=str, default="PongDeterministic-v3",
                    help="Environment id")
parser.add_argument('-l', '--log-dir', type=str, default="/tmp/pong",
                    help="Log directory path")
parser.add_argument('--visualise', action='store_true',
                    help="Visualise the gym environment by running env.render() between each timestep")

def main():
    multiprocessing.set_start_method('spawn')
    args = parser.parse_args()
    env = lambda: create_atari_env(args.env_id)
    env0 = env()
    model = Model(env0.observation_space.shape, env0.action_space.n,
                  is_cuda=True)
    terminate = multiprocessing.Value(ctypes.c_bool, False)
    global_steps = multiprocessing.Value(ctypes.c_int64, 0)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    a3c = A3C(environments=[env() for _ in range(args.num_envs)],
              model=model,
              log_dir=args.log_dir,
              terminate=terminate,
              global_steps=global_steps,
              optimizer=optimizer)
    visualizer = Visualizer(env=env0,
                            model=model,
                            terminate=terminate)
    a3c.load_checkpoint()
    a3c.global_model.share_memory()

    workers = [multiprocessing.Process(target=a3c.run, kwargs={'optimizer': optimizer,
                                                               'worker_id': i})
               for i in range(args.num_workers)]
    visualizer_process = multiprocessing.Process(target=visualizer.run)
    for w in workers:
        w.start()
    visualizer_process.start()
    try:
        while True:
            time.sleep(60)
            a3c.save_checkpoint()
        # for w in workers:
        #     w.join()
        # visualizer_process.join()
    except KeyboardInterrupt:
        terminate.value = True
        for w in workers:
            w.join()
        visualizer_process.join()


if __name__ == '__main__':
    main()