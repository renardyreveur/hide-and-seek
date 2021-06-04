import math
import random

import numpy as np
from typing import Tuple


class Agent:
    def __init__(self, uid: int, agt_class: int, size: int, max_speed: float, max_stamina: float,
                 accel_limit: Tuple[float, float], visual_scope: Tuple[int, float], comm_limit: int):
        # Fixed attributes
        self.uid = uid
        self.max_speed = max_speed
        self.max_stamina = max_stamina
        self.accel_limit = accel_limit
        self.scope = visual_scope                   # (angular scope (0-360), linear scope)
        self.size = size
        self.agt_class = agt_class                  # 2 - Hider, 3 - Seeker
        assert self.agt_class in [2, 3]

        # Dynamic attributes
        self.vision = ((0, 0), np.zeros((2, self.scope[1])))  # First row - vision, Second row - distance
        self.sound = []  # Length = Number of agents around me
        self.comm = []  # Length = Number of agents that sent a signal
        self.comm_limit = comm_limit  # Number of communication events an agent can invoke in it's lifetime
        self.comm_count = 0

        # State history
        self.history = []

        # --- Movement State ---
        self.angle = random.randint(0, 360) * math.pi / 180

        self.speed = 0
        self.stamina = max_stamina

    def action(self):
        """
        At every time step an action is performed.
        The action is one of three {1: "move", 2: "tag", 3: "communicate"}

        Given agent state, return best action based on policy

        :return: action choice, action parameters
        """
        # --- Ruled Based Test Policy ---
        # Stay still just send communication event
        if self.uid == 0:
            if random.choice(list(range(50))) == 1 and self.comm_count < self.comm_limit:
                action = 3
                action_param = {}
                self.comm_count += 1
            else:
                action = 1
                action_param = {"ang_accel": (0 * math.pi / 180), "accel": 0}
            return action, action_param

        # Others
        # If wall in vision, rotate
        vision_array = self.vision[1]
        if 1 in vision_array[0]:
            accel = -1 if self.speed > 0 else 0
            action = 1
            action_param = {"ang_accel": (random.randint(20, 45) * math.pi / 180), "accel": accel}

        # If hider in front, tag
        elif self.agt_class == 3 and 2 in vision_array[0] and vision_array[1][list(vision_array[0]).index(2)] < 60:
            action = 2
            action_param = {}

        # Randomly invoked communication event
        # elif random.choice(list(range(50))) == 1 and self.comm_count < self.comm_limit:
        #     action = 3
        #     action_param = {}
        #     self.comm_count += 1

        # If communication received head towards nearest comm. agent for three steps
        elif len(self.comm) > 0:
            closest_agent = min(self.comm, key=lambda x: x[0])

            # Calculate target angle to the event sender
            target_angle = closest_agent[1] + self.angle
            target_angle = 2*math.pi + target_angle if target_angle < 0 else target_angle
            target_angle = target_angle - 2*math.pi if target_angle > 2*math.pi else target_angle

            # Add target angle to history such that the agent moves until it finds the target angle
            self.history.append(target_angle)
            direction = closest_agent[1]/abs(closest_agent[1])
            action = 1
            action_param = {"ang_accel": direction*math.pi/18, "accel": -1 if self.speed > 0 else 0}

        # If target angle not found, continue searching
        elif len(self.history) > 0:
            direction = self.history[-1]/abs(self.history[-1])
            action = 1
            action_param = {"ang_accel": direction*math.pi/18, "accel": -1 if self.speed > 0 else 0}
            if self.history[-1] - math.pi/9 < self.angle < self.history[-1] + math.pi/9:
                self.history.pop(-1)

        # When there isn't a special event, just move forward
        else:
            st_rate = self.stamina/self.max_stamina
            if st_rate > 0.75:
                accel = np.random.normal(3, 1, 1)
            elif st_rate > 0.4:
                accel = np.random.randint(-1, 3)
            else:
                accel = -1
            action = 1
            action_param = {"ang_accel": (0 * math.pi / 180), "accel": accel}

        return action, action_param
