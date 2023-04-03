# Standard imports
from math import cos, sin, pi

# Full imports
import cv2

# Partial imports
from gymnasium import spaces
from typing import Callable, Tuple

# Aliased imports
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym


class ContiousTelesketch(gym.Env):
    # Define metadata
    metadata = {"render_modes": ["image"]}

    #! MAIN METHODS

    # Constructor
    def __init__(
        self,
        img: np.ndarray,
        diff_function: Callable[[np.array, np.array], float],
        bail_function: Callable[[np.array, np.array], bool],
        reward_function: Callable[[float], float]=None,
        range_segment: Tuple[float, float] = (5, 10),
        thickness_segment: int=5,
        patch_size: Tuple[int, int]=(5, 5),
        change_threshold = 1e-3,
        render_mode="image"
    ) -> None:
              
        # Init variables
        self.img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        self.cnv = np.full(self.img.shape, 1.0, dtype=np.float32)
        self.diff_function = diff_function
        self.bail_function = bail_function
        self.reward_function = reward_function
        self.thickness_segment = thickness_segment
        self.render_mode = render_mode
        self.patch_size = patch_size
        self.range_segment = range_segment
        self.change_threshold = change_threshold


        # Define aux vars
        # Store old diff to avoid recomputing 
        self._done = False
        self._old_diff = self.diff_function(self.img, self.cnv)
        self._reset_flag = False

        # Define environment bounds
        self.action_space = spaces.Box(
            low=np.array([np.finfo(np.float32).min, range_segment[0]]), 
            high=np.array([np.finfo(np.float32).max, range_segment[1]]), 
            shape=(2,), 
            dtype=np.float32
        )

        self.observation_space = spaces.Dict(
            {
                "location": spaces.Box(np.array((0, 0)), np.array(self.img.shape), shape=(2,), dtype=np.float32),
                "dmap": spaces.Box(0, np.sqrt((np.array(self.img.shape) ** 2).sum() ** 0.5), shape=self.img.shape, dtype=np.float32),
                "img": spaces.Box(0, 1, shape=self.img.shape, dtype=np.float32),
                "cnv": spaces.Box(0, 1, shape=self.img.shape, dtype=np.float32),
                "img_patch": spaces.Box(0, 1, shape=self.patch_size , dtype=np.float32),
                "cnv_patch": spaces.Box(0, 1, shape=self.patch_size, dtype=np.float32)
            }
        )

        # Assert correct render mode is selected
        assert render_mode is None or render_mode in self.metadata["render_modes"], "{render_mode} not supported"
        self.render_mode = render_mode

        self.window = None
        self.clock = None
    
    def reset(self, location: Tuple[int, int]=(0, 0), seed=None):
        super().reset(seed=seed)

        # Raise if we provide an invalid location
        assert self.observation_space["location"].contains(np.array(location, dtype=np.float32)), "Location is out-of-bounds"

        # Set main variables
        self.location = np.array(location)
        self.cnv = np.full(self.img.shape, 1.0, dtype=np.float32)

        # Set monitoring variables
        self.no_change_cnt = 0
        self.step_cnt = 0
        self._done = False
        self._old_diff = self.diff_function(self.img, self.cnv)

        # Enable flag
        self._reset_flag = True

        return self._get_obs()

    def step(self, alpha: float, length: float):
        # Asserts
        assert self._reset_flag, "reset method must be called before stepping"
        assert self.range_segment[0] <= length <= self.range_segment[1], f"length if outside speficied range [{str(self.range_segment)[1:-1]}]"
        assert not self._done, "environment has finished, please reset"

        # Compute target x, y
        # Use a minus because our coordinates are inverted on Y axis
        next_location = self.location + np.round(np.array([cos(alpha), -sin(alpha)]) * length).astype(int)
        next_location = np.clip(next_location, [0, 0], self.cnv.shape)

        # Compute new canvas
        next_cnv = cv2.line(self.cnv.copy(), self.location, next_location, (0, 0, 0), self.thickness_segment)

        # Compute reward
        reward = self._get_reward(next_cnv)

        # Update state
        self.cnv = next_cnv
        self.location = next_location
        self.step_cnt += 1

        # Check if we're done
        self._done = self.bail_function(self.img, self.cnv)

        return self._get_obs(), reward, self._done

    def render(self, *args, **kwargs):
        if self.render_mode == "image":
            self._plt_render(*args, **kwargs)


    #! AUX METHODS

    def _get_reward(self, next_cnv):
        # Compute new diff
        new_diff = self.diff_function(self.img, next_cnv)

        print(new_diff)

        # Change in diff
        delta_diff = self._old_diff - new_diff

        # Monitor state
        self.no_change_cnt += 1 if abs(delta_diff) < 1e-3 else 0

        # Update state
        self._old_diff = new_diff

        # Compute reward
        return delta_diff if self.reward_function is None else self.reward_function(delta_diff)

    def _get_dmap(self):
        px, py = self.location.tolist()
        r, c = np.indices(self.cnv.shape)

        return np.sqrt((c - px) ** 2 + (r - py) ** 2) / (np.array(self.cnv.shape) ** 2).sum() ** 0.5
    
    def _get_obs(self):
        return {
            "dmap": self._get_dmap(), 
            "img": self.img, 
            "cnv": self.cnv
        }

    def _plt_render(self, figsize=(5, 5)):
        obs = self._get_obs()
        fig, ax = plt.subplots(1, len(obs), figsize=figsize)

        # Define palette
        palette = np.array([[  0,   0,   0],
                            [255, 255, 255]])
        
        plot_data = zip(
            ("viridis", "gray", "gray"),
            ("Distance Map", "Ref. Img.", "Env. Canvas"), 
            obs.values())
    
        for i, (cmap, title, data) in enumerate(plot_data):
            data = palette[data.astype(np.int32)] if cmap == "gray" else data
            ax[i].imshow(data, cmap=cmap)
            ax[i].set_title(title)

        plt.show()










        
