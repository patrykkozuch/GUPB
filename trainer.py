from locale import normalize
from threading import Event, Thread
from typing import Dict

import torch as th
import gymnasium as gym
import numpy as np
import stable_baselines3.common.vec_env.dummy_vec_env
from gymnasium import spaces
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.preprocessing import is_image_space, get_flattened_obs_dim
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import TensorDict
from stable_baselines3.common.vec_env import VecFrameStack, VecNormalize, VecTransposeImage
from torch import nn

from gupb.__main__ import main

from gupb.controller.bupg.bupg import BUPGController

FACINGS = [
    (0, 1),
    (-1, 0),
    (1, 0),
    (0, -1),
]


class GUPBEnv(gym.Env):
    turn_event = Event()
    game_started = Event()
    train_started = Event()
    action = None

    def __init__(self, ):
        self.action_space = gym.spaces.Box(low=0, high=7, shape=(1,), dtype=np.ushort)
        self.observation_space = gym.spaces.Dict(
            {
                "health": gym.spaces.Box(low=0, high=100, shape=(1,), dtype=np.int16),
                "weapon": gym.spaces.Box(low=0, high=100, shape=(1,), dtype=np.int16),
                "facing": gym.spaces.Box(low=0, high=4, shape=(1,), dtype=np.int16),
                "enemy_in_range": gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.int16),
                "neighborhood": gym.spaces.Box(low=0, high=255, shape=(15, 15, 3), dtype=np.uint8),
            }
        )
        self.controller: None | "BUPGController" = None
        self.last_health = 0

    def attach_controller(self, controller: "BUPGController"):
        self.controller = controller

    def reset(self, seed=None, options=None):
        # Wait for the game to start
        self.game_started.wait()

        self.last_health = self.controller.health
        return {
            "health": self.controller.health,
            "weapon": self.controller.WEAPON_PRIORITY.index(self.controller.weapon.name),
            "enemy_in_range": self.controller.enemy_in_range,
            "facing": FACINGS.index(self.controller.facing.value),
            "neighborhood": self.controller.neighborhood
        }, {}

    def step(self, action):
        # Wait for the controller to take its turn
        self.turn_event.wait()
        self.turn_event.clear()

        observation = {
            "health": self.controller.health,
            "weapon": self.controller.WEAPON_PRIORITY.index(self.controller.weapon.name),
            "enemy_in_range": self.controller.enemy_in_range,
            "facing": FACINGS.index(self.controller.facing.value),
            "neighborhood": self.controller.neighborhood
        }

        # Check if the controller is dead
        if self.controller.died:
            self.controller.train_step.set()
            return observation, self.controller.score, True, False, {"score": self.controller.score}

        reward = self.controller.health - self.last_health
        done = False
        info = {}

        self.last_health = self.controller.health
        self.action = int(action[0])
        # Notify the controller that the training step is over
        self.controller.train_step.set()
        return observation, reward, done, False, info

    def run(self):
        # Assign the environment to the controller
        BUPGController.assign_env(self)
        main(prog_name='python -m gupb')


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
    def __init__(self):
        super(ScoreCallback, self).__init__()
        self.scores = MovingAverage(20)

    def _on_step(self) -> bool:
        done = self.locals["dones"][0]

        if done and "score" in self.locals["infos"][0]:
            score = self.locals["infos"][0]["score"]
            self.logger.record("round/score", self.scores.next(score))

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
            features_dim: int = 512,
            normalized_image: bool = False,
    ) -> None:
        assert isinstance(observation_space, spaces.Box), (
            "NatureCNN must be used with a gym.spaces.Box ",
            f"observation space, not {observation_space}",
        )
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        assert is_image_space(observation_space, check_channels=False, normalized_image=normalized_image), (
            "You should use NatureCNN "
            f"only with images not with {observation_space}\n"
            "(you are probably using `CnnPolicy` instead of `MlpPolicy` or `MultiInputPolicy`)\n"
            "If you are using a custom environment,\n"
            "please check it using our env checker:\n"
            "https://stable-baselines3.readthedocs.io/en/master/common/env_checker.html.\n"
            "If you are using `VecNormalize` or already normalized channel-first images "
            "you should pass `normalize_images=False`: \n"
            "https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html"
        )
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(th.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))


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
            cnn_output_dim: int = 256,
            normalized_image: bool = False,
    ) -> None:
        # TODO we do not know features-dim here before going over all the items, so put something there. This is dirty!
        super().__init__(observation_space, features_dim=1)

        extractors: Dict[str, nn.Module] = {}

        total_concat_size = 0
        for key, subspace in observation_space.spaces.items():
            if is_image_space(subspace, normalized_image=normalized_image):
                extractors[key] = CustomCNN(subspace, features_dim=cnn_output_dim, normalized_image=normalized_image)
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
    env = GUPBEnv()
    vec_env = stable_baselines3.common.vec_env.dummy_vec_env.DummyVecEnv([lambda: env])
    vec_env = VecFrameStack(vec_env, n_stack=8)

    # Utwórz model SAC z domyślnym MLP policy
    model = SAC(
        "MultiInputPolicy",
        env,
        gamma=0.975,
        verbose=2,
        tensorboard_log="./logs/",
        policy_kwargs=dict(
            features_extractor_class=CustomCombinedExtractor,
            features_extractor_kwargs=dict(cnn_output_dim=32, normalized_image=False)
            )
        )

    t1 = Thread(target=env.run, daemon=True).start()
    env.train_started.set()

    # Trening modelu przez 100000 kroków
    model.learn(total_timesteps=1000000, callback=[ScoreCallback()])
