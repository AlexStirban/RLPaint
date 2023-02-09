"""Discrete Telesketch environment for gym"""

# Standard imports
from math import cos, sin, pi

# Full imports
import gym
import cv2

# Partial imports
from gym import spaces

# Aliased imports
import numpy as np
import matplotlib.pyplot as plt


class DiscreteTelesketchEnv(gym.Env):
    metadata = {"render_modes": ["image"]}

    def __init__(self, 
                ref_canvas, 
                sim_func, 
                segment_length=5, 
                stroke_thickness=5, 
                patch_size=(25, 25), 
                render_mode=None) -> None:

        # Save ref image & sim func
        self.ref_canvas = self._to_norm_gray(ref_canvas)
        self.sim_func = sim_func
        self._patch_size = patch_size
        self._reset_called = False

        # Save segment params
        self.segment_length = segment_length
        self.stroke_thickness = stroke_thickness
        
        # Define action space
        self.action_space = spaces.Discrete(8)

        # Define observation space
        self.observation_space = spaces.Dict(
            {
                "loc": spaces.Box(np.array((0, 0)), np.array(self.ref_canvas.shape), dtype=int),
                "ref": spaces.Box(0, 1, shape=self.ref_canvas.shape, dtype=np.float32),
                "cnv": spaces.Box(0, 1, shape=self.ref_canvas.shape, dtype=np.float32),
                "ref_patch": spaces.Box(0, 1, shape=patch_size, dtype=np.float32),
                "cnv_patch": spaces.Box(0, 1, shape=patch_size, dtype=np.float32)
            }
        )

        # Assert correct render mode is selected
        assert render_mode is None or render_mode in self.metadata["render_modes"], "{render_mode} not supported"
        self.render_mode = render_mode

        self.window = None
        self.clock = None

    def _to_norm_gray(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img.astype(np.float32) / 255

    def _get_obs(self):
        cnv_patch = self._compute_patch(self._canvas, self._loc, self._patch_size)
        ref_patch = self._compute_patch(self.ref_canvas, self._loc, self._patch_size)

        return {"loc": self._loc, 
                "ref": self.ref_canvas, 
                "cnv": self._canvas,
                "ref_patch": ref_patch,
                "cnv_patch": cnv_patch
        }
    
    def _get_info(self):
        return {"sim": self.sim_func(self._canvas, self.ref_canvas)}

    def _compute_rewards(self, ref_canvas, new_canvas, old_canvas):
        old_sim = self.sim_func(ref_canvas, old_canvas)
        new_sim = self.sim_func(ref_canvas, new_canvas)

        return 1 if new_sim < old_sim else 0
    
    def _compute_patch(self, canvas: np.ndarray, loc: np.ndarray, size: np.ndarray) -> np.ndarray:
        # Define square patch
        xmin = np.clip(int(loc[0] - np.floor(size[0] / 2)), 0, canvas.shape[0])
        xmax = np.clip(int(loc[0] + np.floor(size[0] / 2)), 0, canvas.shape[0])
        ymin = np.clip(int(loc[1] - np.floor(size[0] / 2)), 0, canvas.shape[1])
        ymax = np.clip(int(loc[1] + np.floor(size[0] / 2)), 0, canvas.shape[1])

        # Get patch
        img = canvas[xmin:xmax, ymin:ymax]
        padding = [
            # Top, bottom
            (0, size[1] - img.shape[1]),
            # Right, left
            (0, size[0] - img.shape[0])
        ]
        
        return np.pad(img, padding, mode="constant", constant_values=1)

    def reset(self, loc=(0, 0), seed=None,):
        super().reset(seed=seed)

        self._loc = np.array(loc)
        self._canvas = np.full(self.ref_canvas.shape, 1, dtype=np.float32)
        
        # Reset flag
        self._reset_called = True

        return self._get_obs(), self._get_info()
    
    def step(self, action):
        # Check action is correct and reset has been called
        assert action >= 0 and action < self.action_space.n, "Action is not valid"
        assert self._reset_called, "Reset must be called before step"

        # Compute target x, y
        x = cos(action * pi / 4) * self.segment_length
        y = sin(action * pi / 4) * self.segment_length

        # Compute new states
        new_loc = self._loc + np.round(np.array([x, -y])).astype(int)
        new_canvas = cv2.line(self._canvas.copy(), self._loc, new_loc, (0, 0, 0), self.stroke_thickness)

        # Compute rewards
        reward = self._compute_rewards(self.ref_canvas, new_canvas, self._canvas)

        # Update state
        self._loc = new_loc
        self._canvas = new_canvas

        return self._get_obs(), reward, False, self._get_info()

    def render(self):
        if self.render_mode == "image":
            self._plt_render()

    def _plt_render(self):
        fig, ax = plt.subplots(2, 2, figsize=(8, 8))
        obs = self._get_obs()

        ref = cv2.cvtColor((obs["ref"] * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        cnv = cv2.cvtColor((obs["cnv"] * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        ref_patch = cv2.cvtColor((obs["ref_patch"] * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        cnv_patch = cv2.cvtColor((obs["cnv_patch"] * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

        ax[0][0].imshow(ref, cmap="gray")
        ax[0][1].imshow(cnv, cmap="gray")
        ax[1][0].imshow(ref_patch, cmap="gray")
        ax[1][1].imshow(cnv_patch, cmap="gray")

        ax[0][0].set_title("Ref. Canvas")
        ax[0][1].set_title("Env. Canvas")
        ax[1][0].set_title("Ref. Patch Canvas")
        ax[1][1].set_title("Env. Patch Canvas")

        plt.show()
