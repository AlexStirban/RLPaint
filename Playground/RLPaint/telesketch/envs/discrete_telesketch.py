"""Discrete Telesketch environment for gym"""

# Standard imports
from math import cos, sin, pi

# Full imports
import gymnasium as gym
import cv2

# Partial imports
from gymnasium import spaces
from numpy import linalg as LA

# Aliased imports
import numpy as np
import matplotlib.pyplot as plt


class DiscreteTelesketchEnv(gym.Env):
    metadata = {"render_modes": ["image"]}

    def __init__(self, 
                ref_canvas, 
                sim_func,
                condition=None,
                segment_length=5, 
                stroke_thickness=5, 
                patch_size=(25, 25),
                factor=1,
                render_mode=None) -> None:

        # Save ref image & sim func
        self.ref_canvas = self._to_norm_gray(ref_canvas)
        self.sim_func = sim_func
        self._patch_size = patch_size
        self._reset_called = False
        self._condition = condition
        self._factor = factor

        # Save segment params
        self.segment_length = segment_length
        self.stroke_thickness = stroke_thickness
        
        # Define action space
        self.action_space = spaces.Discrete(8)

        # Define observation space
        self.observation_space = spaces.Dict(
            {
                "loc": spaces.Box(0, 1, shape=self.ref_canvas.shape, dtype=np.float32),
                "ref": spaces.Box(0, 1, shape=self.ref_canvas.shape, dtype=np.float32),
                "cnv": spaces.Box(0, 1, shape=self.ref_canvas.shape, dtype=np.float32),
                "ref_patch": spaces.Box(0, 1, shape=patch_size, dtype=np.float32),
                "cnv_patch": spaces.Box(0, 1, shape=patch_size, dtype=np.float32)
            }
        )

        self._step_counter = 0
        self._stuck_counter = 0

        # Assert correct render mode is selected
        assert render_mode is None or render_mode in self.metadata["render_modes"], "{render_mode} not supported"
        self.render_mode = render_mode

        self.window = None
        self.clock = None

    def _get_distance_map(self):
        px, py = self._loc[0], self._loc[1]
        r, c = np.indices(self.ref_canvas.shape)

        return np.sqrt((c - px) ** 2 + (r - py) ** 2) / self.ref_canvas.shape[0]

    def _to_norm_gray(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img.astype(np.float32) / 255

    def _get_obs(self):
        cnv_patch = self._compute_patch(self._canvas, self._loc, self._patch_size)
        ref_patch = self._compute_patch(self.ref_canvas, self._loc, self._patch_size)
        d_map = self._get_distance_map()

        return {"loc": d_map.astype(np.float32), 
                "ref": self.ref_canvas.astype(np.float32), 
                "cnv": self._canvas.astype(np.float32),
                "ref_patch": ref_patch.astype(np.float32),
                "cnv_patch": cnv_patch.astype(np.float32)
        }
    
    def _get_info(self):
        return {"sim": self.sim_func(self.ref_canvas, self._canvas).item(0)}

    def _compute_rewards(self, ref_canvas, new_canvas, old_canvas, new_loc):
        old_diff = self.sim_func(ref_canvas, old_canvas)
        new_diff = self.sim_func(ref_canvas, new_canvas)

        if abs(old_diff - new_diff) < 1e-3:
            self._stuck_counter += 1
            return -1

        # Penalize not moving
        diff = (old_diff - new_diff).item() * 10
        return diff if diff > 0 else diff * 0.4 
    
    def _compute_patch(self, canvas: np.ndarray, loc: np.ndarray, size: np.ndarray) -> np.ndarray:
        # Define square patch

        xmin = loc[0] - size[0] // 2 - size[0] % 2
        xmax = loc[0] + size[0] // 2
        ymin = loc[1] - size[1] // 2 - size[1] % 2
        ymax = loc[1] + size[1] // 2

        # Bound coords
        if xmin < 0:
            xmax += 0 - xmin
            xmin = 0
        elif xmax > canvas.shape[0]:
            xmin -= xmax - canvas.shape[0]
            xmax = canvas.shape[0]

        if ymin < 0:
            ymax += 0 - ymin
            ymin = 0
        elif ymax > canvas.shape[1]:
            ymin -= ymax - canvas.shape[1]
            ymax = canvas.shape[1]

        # Get patch
        img = canvas[ymin:ymax, xmin:xmax]

        return img

    def reset(self, loc=(0, 0), seed=None,):
        super().reset(seed=seed)

        self._step_counter = 0
        self._stuck_counter = 0
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
        x = cos(-action * pi / 4) * self.segment_length
        y = sin(-action * pi / 4) * self.segment_length

        # Compute new states
        new_loc = self._loc + np.round(np.array([x, -y])).astype(int)
        new_loc = np.clip(new_loc, [0, 0], self.ref_canvas.shape)
        new_canvas = cv2.line(self._canvas.copy(), self._loc, new_loc, (0, 0, 0), self.stroke_thickness)

        # Compute rewards
        reward = self._compute_rewards(self.ref_canvas, new_canvas, self._canvas, new_loc)

        # Update state
        self._loc = new_loc
        self._canvas = new_canvas
        self._step_counter += 1

        
        done = False

        # Check if we're close to the actual image
        if self._condition is not None:
            done = self._condition(self.ref_canvas, new_canvas)

            if done:
                reward = 100
        
        # Check if we're stuck
        if self._stuck_counter > 20:
            done = True
            reward = -100

        return self._get_obs(), reward, done, self._get_info()

    def render(self):
        if self.render_mode == "image":
            self._plt_render()

    def _plt_render(self):
        fig, ax = plt.subplots(2, 3, figsize=(8, 8))
        obs = self._get_obs()
        
        dmap = obs["loc"]
        ref = cv2.cvtColor((obs["ref"] * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        cnv = cv2.cvtColor((obs["cnv"] * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        ref_patch = cv2.cvtColor((obs["ref_patch"] * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        cnv_patch = cv2.cvtColor((obs["cnv_patch"] * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

        ax[0][0].imshow(ref, cmap="gray")
        ax[0][1].imshow(cnv, cmap="gray")
        ax[1][0].imshow(ref_patch, cmap="gray")
        ax[1][1].imshow(cnv_patch, cmap="gray")
        ax[0][2].imshow(dmap)
        ax[1][2].axis("off")

        ax[0][0].set_title("Ref. Canvas")
        ax[0][1].set_title("Env. Canvas")
        ax[1][0].set_title("Ref. Patch Canvas")
        ax[1][1].set_title("Env. Patch Canvas")
        ax[0][2].set_title("Distance Map")

        plt.show()
