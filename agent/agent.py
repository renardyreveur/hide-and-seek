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
        print(self.angle)
        """
        At every time step an action is performed.
        The action is one of three {1: "move", 2: "tag", 3: "communicate"}

        Given agent state, return best action based on policy

        :return: action choice, action parameters
        """
        # --- Ruled Based Test Policy ---

        if self.uid == 0:
            if random.choice(list(range(50))) == 1 and self.comm_count < self.comm_limit:
                action = 3
                action_param = {}
                self.comm_count += 1
            else:
                action = 1
                action_param = {"ang_accel": (0 * math.pi / 180), "accel": 0}
            return action, action_param

        vision_array = self.vision[1]

        # If wall in vision, rotate
        if 1 in vision_array[0]:
            action = 1
            action_param = {"ang_accel": (random.randint(20, 45) * math.pi / 180), "accel": -10}

        # If stationary for 3 time steps rotate
        # elif len(set(self.history)) == 1:
        #     action = 1
        #     action_param = {"ang_accel": (random.randint(20, 45) * math.pi / 180), "accel": 0}

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
            print(closest_agent[1])
            self.history.append(closest_agent[1] - self.angle)
            direction = closest_agent[1]/abs(closest_agent[1])
            action = 1
            action_param = {"ang_accel": direction*math.pi/18, "accel": -1}

        elif len(self.history) > 0:
            print("HEHEHEHEHE")
            direction = self.history[-1]/abs(self.history[-1])
            action = 1
            action_param = {"ang_accel": direction*math.pi/18, "accel": -1}
            print(self.history[-1] - math.pi/9, self.history[-1] + math.pi/9, self.angle)
            if self.history[-1] - math.pi/9 < self.angle < self.history[-1] + math.pi/9:
                self.history.pop(-1)

        # When there isn't a special event, just move forward
        else:
            action = 1
            action_param = {"ang_accel": (0 * math.pi / 180), "accel": 5}

        return action, action_param
