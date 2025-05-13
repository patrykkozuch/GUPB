import time
from threading import Event, Thread
from typing import Dict

import cv2
import torch as th
import gymnasium as gym
import numpy as np
import stable_baselines3.common.vec_env.dummy_vec_env
from gymnasium import spaces
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.preprocessing import get_flattened_obs_dim
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import TensorDict
from stable_baselines3.common.vec_env import VecNormalize, VecFrameStack

from torch import nn

from gupb.__main__ import main

from gupb.controller.bupg.bupg import BUPGController
from gupb.controller.bupg.utils import position_change_to_move
from gupb.model.characters import CHAMPION_STARTING_HP, Action
from gupb.model.coordinates import Coords
from gupb.model.games import Game

FACINGS = [
    (0, 1),
    (-1, 0),
    (1, 0),
    (0, -1),
]

directions = {
    "up": Coords(0, -1),
    "down": Coords(0, 1),
    "left": Coords(-1, 0),
    "right": Coords(1, 0)
}


class GUPBEnv(gym.Env):
    def __init__(self, idx: int = 0):
        self.env_id = idx
        self.turn_event = Event()
        self.game_started = Event()
        self.train_started = Event()
        self.action = None
        self.env_thread = None

        self.action_space = gym.spaces.Discrete(7)
        self.observation_space = gym.spaces.Dict(
            {
                "health": gym.spaces.Box(low=0, high=4, shape=(1,), dtype=np.float64),
                "weapon": gym.spaces.Box(low=0, high=1, shape=(8,), dtype=np.int64),
                "facing": gym.spaces.Box(low=0, high=1, shape=(4,), dtype=np.int64),
                "enemy_in_range": gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.int64),
                "distance_to_menhir": gym.spaces.Box(low=0, high=10000, shape=(1,), dtype=np.float64),
                "menhir_location": gym.spaces.Box(low=-1000, high=1000, shape=(2,), dtype=np.int64),
                "mist_ratio": gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float64),
                "seen": gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float64),
                "players_alive": gym.spaces.Box(low=0, high=20, shape=(1,), dtype=np.int64),
                "map": gym.spaces.Box(low=-1, high=1, shape=(9, 11, 11), dtype=np.float64)
            }
        )
        self.controller: None | "BUPGController" = None
        self.last_health = 0
        self.last_seen = 0
        self.steps = 0
        self.num_players = 0
        self.attacked = False
        self.mist_nearby = False

    def attach_controller(self, controller: "BUPGController"):
        self.controller = controller

    def _get_obs(self):
        weapon = [0 for _ in range(8)]
        weapon[self.controller.WEAPON_PRIORITY.index(self.controller.weapon.name)] = 1

        facings = [0 for _ in range(4)]
        facings[FACINGS.index(self.controller.facing.value)] = 1

        map = self.controller.maps

        if self.controller.map_knowledge.menhir_location is not None:
            menhir_location = np.array(self.controller.map_knowledge.menhir_location) - np.array(
                self.controller.position
            )
        else:
            menhir_location = np.array([0, 0])

        return {
            "health": np.array([self.controller.health / 20]),
            "weapon": np.array(weapon),
            "seen": np.array([self.controller.map_knowledge.seen()]),
            "facing": np.array(facings),
            "enemy_in_range": np.array([int(self.controller.enemy_in_range)]),
            "distance_to_menhir": np.array([self.controller.distance_to_menhir()]),
            "menhir_location": menhir_location,
            "mist_ratio": np.array([self.controller.map_knowledge.mist_ratio()]),
            "players_alive": np.array([self.controller.knowledge.no_of_champions_alive]),
            "map": map
        }

    def reset(self, seed=None, options=None):
        # Wait for the game to start
        self.game_started.wait()
        self.steps = 0
        self.attacked = False
        self.last_health = CHAMPION_STARTING_HP
        obs = self._get_obs()
        obs["health"] = np.array([CHAMPION_STARTING_HP / 20])
        self.last_seen = obs["seen"][0]
        self.num_players = self.controller.knowledge.no_of_champions_alive
        gen = Game._fibonacci()
        self.last_score = [next(gen) for _ in range(self.num_players)][-1]
        return obs, {}

    def step(self, action):
        # Wait for the controller to take its turn
        self.turn_event.wait()
        self.turn_event.clear()

        observation = self._get_obs()
        self.steps += 1

        # Check if the controller is dead
        if self.controller.died:
            self.controller.train_step.set()
            observation["health"] = np.array([0])
            win = self.controller.score == self.last_score
            return observation, int(win), True, False, {"score": self.controller.score, "won": win}

        info = {}

        reward = 0.001

        lost_health = self.controller.health - self.last_health

        if self.mist_nearby and self.controller.map_knowledge.mist[self.controller.position[1], self.controller.position[0]] < 0.5:
            reward += 0.010
            info["run_from_mist"] = True

        if self.attacked:
            self.attacked = False
            if not lost_health and self.controller.weapon.name != "bow_unloaded":
                    info["attacked_enemy"] = True
                    reward += 0.005

        if lost_health < 0:
            info["damaged"] = True
            reward += lost_health / 100
        elif lost_health > 0:
            info["healed"] = True
            reward += lost_health / 100

        neighbor_cells = [self.controller.position + direction for direction in directions.values()]
        self.mist_nearby = any([self.controller.map_knowledge.mist[nc[1], nc[0]] > 0.5 for nc in neighbor_cells])

        done = False

        seen_reward = max(0, observation["seen"][0] - self.last_seen)
        reward += seen_reward

        self.last_seen = observation["seen"][0]
        self.last_health = self.controller.health
        self.action = action

        if action == 6:
            self.attacked = True
            info["ATTACK!"] = True

        info["seen"] = observation["seen"]

        # Notify the controller that the training step is over
        self.controller.train_step.set()
        return observation, reward, done, False, info

    def run(self):
        # Assign the environment to the controller
        BUPGController.assign_env(self)
        config = "gupb/default_config_vis.py" if self.env_id == 0 else "gupb/default_config.py"
        main.main(args=['-c', config])

    def start_training(self):
        # Notify the controller that the training has started
        self.train_started.set()

    def start_thread(self):
        # Create a thread to run the environment
        self.env_thread = Thread(target=self.run, daemon=True)
        self.env_thread.start()

    def action_masks(self):
        actions = list(Action)
        mask = [True for _ in range(7)]

        movement = {}
        action_indices = {}

        for direction, offset in directions.items():
            pos = self.controller.position + offset

            action = position_change_to_move(self.controller.position, pos, self.controller.facing, "xy")
            action_indices[direction] = actions.index(action)
            movement[direction] = (
                    pos in self.controller.map_knowledge.terrain and
                    self.controller.map_knowledge.terrain[pos].description().type in ["forest", "land"]
            )

        mask[0] = True
        mask[1] = True

        mask[-1] = self.controller.enemy_in_range or self.controller.weapon.name == "bow_unloaded"

        for direction in directions:
            mask[action_indices[direction]] = movement[direction]

        return mask


class MovingAverage:
    def __init__(self, window_size):
        self.window_size = window_size
        self.sum = 0
        self.count = 0
        self.queue = []

    def next(self, val):
        if self.count == self.window_size:
            self.sum -= self.queue.pop(0)
        else:
            self.count += 1

        self.queue.append(val)
        self.sum += val
        return self.sum / self.count


class ScoreCallback(BaseCallback):
    def __init__(self, num_envs: int):
        super(ScoreCallback, self).__init__()
        self.scores = MovingAverage(100)
        self.steps_near_menhir = MovingAverage(100)
        self.times_attacked_enemy = MovingAverage(100)
        self.times_damaged = MovingAverage(100)
        self.times_healed = MovingAverage(100)
        self.times_attacked = MovingAverage(100)
        self.run_from_mist = MovingAverage(100)
        self.won = MovingAverage(100)
        self.exploration_rate = MovingAverage(100)
        self.env_stats = {i: {"near_menhir": 0, "attacked_enemy": 0, "damaged": 0, "healed": 0, "attack": 0, "seen": 0, "run_from_mist": 0, "won": 0}
                          for i
                          in range(num_envs)}

    def _on_step(self) -> bool:
        dones = self.locals["dones"]
        infos = self.locals["infos"]
        for idx, done in enumerate(dones):
            info = infos[idx]
            if info.get("near_menhir", False):
                self.env_stats[idx]["near_menhir"] += 1
            if info.get("attacked_enemy", False):
                self.env_stats[idx]["attacked_enemy"] += 1
            if info.get("damaged", False):
                self.env_stats[idx]["damaged"] += 1
            if info.get("healed", False):
                self.env_stats[idx]["healed"] += 1
            if info.get("ATTACK!", False):
                self.env_stats[idx]["attack"] += 1
            if info.get("seen", False):
                self.env_stats[idx]["seen"] = info.get("seen")[0]
            if info.get("run_from_mist", False):
                self.env_stats[idx]["run_from_mist"] += 1

            if done:
                self.logger.record("round/score", self.scores.next(info["score"]))
                self.logger.record("round/won", self.won.next(info["won"]))

                self.logger.record(
                    "round/steps_near_menhir",
                    self.steps_near_menhir.next(self.env_stats[idx]["near_menhir"])
                )
                self.logger.record(
                    "round/attacked_enemy",
                    self.times_attacked_enemy.next(self.env_stats[idx]["attacked_enemy"])
                )

                self.logger.record("round/times_damaged", self.times_damaged.next(self.env_stats[idx]["damaged"]))
                self.logger.record("round/times_healed", self.times_healed.next(self.env_stats[idx]["healed"]))
                self.logger.record("round/times_attacked", self.times_attacked.next(self.env_stats[idx]["attack"]))
                self.logger.record("round/run_from_mist", self.run_from_mist.next(self.env_stats[idx]["run_from_mist"]))

                if self.env_stats[idx]["seen"]:
                    self.logger.record(
                        "round/seen",
                        self.exploration_rate.next(self.env_stats[idx]["seen"])
                    )

                self.env_stats[idx] = {
                    "near_menhir": 0,
                    "attacked_enemy": 0,
                    "damaged": 0,
                    "healed": 0,
                    "attack": 0,
                    "seen": 0,
                    "run_from_mist": 0,
                    "won": 0
                }
        return True


class CustomCNN(BaseFeaturesExtractor):
    """
    CNN from DQN Nature paper:
        Mnih, Volodymyr, et al.
        "Human-level control through deep reinforcement learning."
        Nature 518.7540 (2015): 529-533.

    :param observation_space:
    :param features_dim: Number of features extracted.
        This corresponds to the number of unit for the last layer.
    :param normalized_image: Whether to assume that the image is already normalized
        or not (this disables dtype and bounds checks): when True, it only checks that
        the space is a Box and has 3 dimensions.
        Otherwise, it checks that it has expected dtype (uint8) and bounds (values in [0, 255]).
    """

    def __init__(
            self,
            observation_space: gym.Space,
            features_dim: int = 512
    ) -> None:
        assert isinstance(observation_space, spaces.Box), (
            "NatureCNN must be used with a gym.spaces.Box ",
            f"observation space, not {observation_space}",
        )
        super().__init__(observation_space, features_dim)

        self.map_size = 11  # MAP_SIZE constant

        n_input_channels = observation_space.shape[0]

        self.cnn = nn.Sequential(
            # Initial 5x5 downsampling
            nn.Conv2d(n_input_channels + 2, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            # Bottleneck block 1
            nn.Conv2d(32, 16, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=1),
            nn.ReLU(),
            # Bottleneck block 2
            nn.Conv2d(32, 16, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=1),
            nn.ReLU(),
            nn.Flatten()
        )

        x_pos_encoding = np.linspace(-1, 1, num=self.map_size)
        y_pos_encoding = np.linspace(-1, 1, num=self.map_size)
        xv, yv = np.meshgrid(x_pos_encoding, y_pos_encoding)
        self.register_buffer("xv_tensor", th.tensor(xv, dtype=th.float32).unsqueeze(0))  # shape: 1x15x15
        self.register_buffer("yv_tensor", th.tensor(yv, dtype=th.float32).unsqueeze(0))  # shape: 1x15x15

        # Compute shape by doing one forward pass
        with th.no_grad():
            sample = th.as_tensor(observation_space.sample()[None]).float()
            sample = self._add_positional_encoding(sample)
            n_flatten = self.cnn(sample).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def _add_positional_encoding(self, x: th.Tensor) -> th.Tensor:
        batch_size = x.shape[0]
        xv_exp = self.xv_tensor.expand(batch_size, -1, -1).unsqueeze(1)  # Bx1x15x15
        yv_exp = self.yv_tensor.expand(batch_size, -1, -1).unsqueeze(1)  # Bx1x15x15
        return th.cat([x, xv_exp, yv_exp], dim=1)  # Concatenate on channel dim

    def forward(self, observations: th.Tensor) -> th.Tensor:
        x = self._add_positional_encoding(observations)
        return self.linear(self.cnn(x))


class CustomCombinedExtractor(BaseFeaturesExtractor):
    """
    Combined features extractor for Dict observation spaces.
    Builds a features extractor for each key of the space. Input from each space
    is fed through a separate submodule (CNN or MLP, depending on input shape),
    the output features are concatenated and fed through additional MLP network ("combined").

    :param observation_space:
    :param cnn_output_dim: Number of features to output from each CNN submodule(s). Defaults to
        256 to avoid exploding network sizes.
    :param normalized_image: Whether to assume that the image is already normalized
        or not (this disables dtype and bounds checks): when True, it only checks that
        the space is a Box and has 3 dimensions.
        Otherwise, it checks that it has expected dtype (uint8) and bounds (values in [0, 255]).
    """

    def __init__(
            self,
            observation_space: spaces.Dict,
            cnn_output_dim: int = 256
    ) -> None:
        # TODO we do not know features-dim here before going over all the items, so put something there. This is dirty!
        super().__init__(observation_space, features_dim=1)

        extractors: Dict[str, nn.Module] = {}

        total_concat_size = 0
        for key, subspace in observation_space.spaces.items():
            if key == "map":
                extractors[key] = CustomCNN(subspace, features_dim=cnn_output_dim)
                total_concat_size += cnn_output_dim
            else:
                # The observation key is a vector, flatten it if needed
                extractors[key] = nn.Flatten()
                total_concat_size += get_flattened_obs_dim(subspace)

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations: TensorDict) -> th.Tensor:
        encoded_tensor_list = []

        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        return th.cat(encoded_tensor_list, dim=1)


if __name__ == "__main__":
    def make_env(idx: int):
        def _init():
            env = GUPBEnv(idx)
            env = Monitor(env)
            return env

        return _init

    vec_env = stable_baselines3.common.vec_env.SubprocVecEnv([make_env(i) for i in range(16)])
    vec_env = VecFrameStack(vec_env, 2, channels_order="first")
    vec_env = VecNormalize(vec_env)

    args = dict(
        n_steps=256,
        batch_size=128,
        ent_coef=0.01,
        verbose=0,
        tensorboard_log="./logs/",
        policy_kwargs=dict(
            features_extractor_class=CustomCombinedExtractor,
            features_extractor_kwargs=dict(cnn_output_dim=64),
        )
    )

    # # Utwórz model SAC z domyślnym MLP policy
    model = MaskablePPO.load("checkpoints/model_1245184_steps.zip", vec_env)

    vec_env.env_method("start_thread")
    vec_env.env_method("start_training")

    checkpoint_callback = CheckpointCallback(
        save_freq=4096,
        save_path="./checkpoints/",
        name_prefix="model",
    )

    # Trening modelu przez 10000000 kroków z callbackami
    model.learn(total_timesteps=10000000, callback=[ScoreCallback(16), checkpoint_callback], reset_num_timesteps=False)
