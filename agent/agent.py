import math
import random

import numpy as np
from typing import Tuple


class Agent:
    def __init__(self, agt_class: int, max_speed: float, max_stamina: float, accel_limit: Tuple[float, float],
                 visual_scope: Tuple[int, float], size: int):
        # Fixed attributes
        self.max_speed = max_speed
        self.max_stamina = max_stamina
        self.accel_limit = accel_limit
        self.scope = visual_scope                 # (angular scope (0-360), linear scope)
        self.size = size
        self.agt_class = agt_class                # 2 - Hider, 3 - Seeker
        assert self.agt_class in [2, 3]

        # Dynamic attributes
        self.vision = np.zeros((2, self.scope[1]))  # First row - vision, Second row - distance
        self.sound = (0, 0)
        self.comm = (0, 0)

        # State history
        self.history = []

        # --- Movement State ---
        self.angle = 0
        self.speed = 0
        self.stamina = max_stamina

    def action(self):
        """
        At every time step an action is performed.
        The action is one of three {1: "move", 2: "tag", 3: "communicate"}

        Given agent state, return best action based on policy

        :return: action choice, action parameters
        """
        # --- Policy ---
        # If wall in vision, rotate
        if 1 in self.vision[0]:
            action = 1
            action_param = {"ang_accel": (random.randint(20, 45) * math.pi / 180),
                            "accel": -10}

        # If stationary for 3 time steps rotate
        elif len(set(self.history)) == 1:
            action = 1
            action_param = {"ang_accel": (random.randint(20, 45) * math.pi / 180),
                            "accel": 0}

        # If hider in front, tag
        elif self.agt_class == 3 and 2 in self.vision[0] and self.vision[1][list(self.vision[0]).index(2)] < 60:
            action = 2
            action_param = {}

        # When there isn't a special event, just move forward
        else:
            action = 1
            action_param = {"ang_accel": (0 * math.pi / 180), "accel": 5}

        return action, action_param
