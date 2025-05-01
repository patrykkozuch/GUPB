from threading import Event, Thread

import gymnasium as gym
import stable_baselines3.common.vec_env.dummy_vec_env
from stable_baselines3 import SAC

from gupb.__main__ import main

from gupb.controller.bupg.bupg import BUPGController


class GUPBEnv(gym.Env):
    turn_event = Event()
    game_started = Event()
    train_started = Event()

    def __init__(self, ):
        self.action_space = gym.spaces.Box(low=0, high=255, shape=(10,), dtype=int)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(10,), dtype=int)
        self.controller: None | "BUPGController" = None

    def attach_controller(self, controller: "BUPGController"):
        self.controller = controller

    def reset(self, seed=None, options=None):
        # Wait for the game to start
        self.game_started.wait()

        return self.observation_space.sample(), {}

    def step(self, action):
        # Wait for the controller to take its turn
        self.turn_event.wait()
        self.turn_event.clear()

        if self.controller.died:
            print("Controller died, skipping step")
            self.controller.train_step.set()
            return self.observation_space.sample(), 0, False, False, {}

        observation = self.observation_space.sample()
        reward = 1.0
        done = False
        info = {}

        # Notify the controller that the training step is over
        self.controller.train_step.set()
        return observation, reward, done, False, info

    def run(self):
        # Assign the environment to the controller
        BUPGController.assign_env(self)
        main(prog_name='python -m gupb')


if __name__ == "__main__":
    env = GUPBEnv()
    vec_env = stable_baselines3.common.vec_env.dummy_vec_env.DummyVecEnv([lambda: env])

    # Utwórz model SAC z domyślnym MLP policy
    model = SAC("MlpPolicy", env, verbose=1)

    t1 = Thread(target=env.run, daemon=True).start()
    env.train_started.set()
    # Trening modelu przez 100000 kroków
    model.learn(total_timesteps=10000)