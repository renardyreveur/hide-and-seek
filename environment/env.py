import math
import sys
from typing import Tuple, Dict

import cv2
import numpy as np

# test_agent is in the environment folder
from agent import Agent
# Get states
from get_states import get_vision, get_sound, get_communication

# If you get a truncated representation, but want the full array, try
np.set_printoptions(threshold=sys.maxsize)


class World:
    def __init__(self, agent_config: Tuple[int, int], agent_kwargs: Dict,
                 map_size: Tuple[int, int], max_num_walls: int, borders: bool, sound_limit: int):
        """
        :param agent_config: Number of Hiders and Seekers
        :param agent_kwargs: Keyword arguments to create agents
        :param map_size: Max Width and Height of the map
        :param max_num_walls: Maximum number of walls to draw on the map
        :param borders: Whether to have borders on the map or not
        """
        # ---- Map configurations ----
        mw, mh = map_size
        self.width, self.height = np.random.randint(mw // 2, mw), np.random.randint(mh // 2, mh)
        self.num_walls = np.random.randint(max_num_walls // 2, max_num_walls)

        # Create empty map (+ borders if requested)
        self.map = np.zeros((self.width, self.height))
        self.borders = borders
        if borders:
            self.map = np.pad(self.map, pad_width=5, mode='constant', constant_values=1.)
            self.width += 10
            self.height += 10

        # Add walls to the map
        self.wall_loc = self.make_walls()
        print(f'Map Size:\nWidth: {self.width}, Height: {self.height}\nNum Walls: {self.num_walls}')
        print("\n Map Legend: {1: walls, 2: hiders, 3: seekers}")

        # ---- Agent configurations ----
        num_hiders, num_seekers = agent_config
        self.agents = [Agent(2, **agent_kwargs) if i < num_hiders else Agent(3, **agent_kwargs)
                       for i in range(num_seekers + num_hiders)]
        self.agent_loc = self.init_agent_loc()

        self.sound_limit = sound_limit
        # TODO: zip agents and agent_loc
        print(f'Agents:\nHiders: {num_hiders}, Seekers: {num_seekers}')

    # Make walls
    def make_walls(self):
        """
        the map is occupied(1) by some walls
        """
        wall_loc = []
        for i in range(self.num_walls):
            # Starting point of the wall
            x0 = np.random.randint(0, self.width)
            y0 = np.random.randint(0, self.height)

            # length of the wall
            w_len = np.random.randint(0, min(self.width - x0, self.height - y0))

            # shape of the wall (horizontal, vertical, 45) - tuple contains a int
            # TODO: Add more 'interesting' cases
            ang = np.random.choice(3, 1, p=[0.3, 0.3, 0.4])

            if ang[0] == 0:  # horizontal
                x1 = x0 + w_len
                y1 = y0
            elif ang[0] == 1:  # vertical
                x1 = x0
                y1 = y0 + w_len
            else:  # 45 degrees slant
                x1 = x0 + w_len
                y1 = y0 + w_len

            # Get wall location coordinates
            xs = np.arange(x0, x1)
            ys = np.arange(y0, y1)
            # if (x0 == x1) or (y0 == y1) then xs, ys -> []
            if x0 == x1:
                xs = [x0] * w_len
            if y0 == y1:
                ys = [y0] * w_len

            # If the pixel is occupied by the wall, let's put 1 there
            for coord in zip(xs, ys):
                self.map[coord] = 1
                wall_loc.append(coord)

        return wall_loc

    # Make list that will hold agent locations
    def init_agent_loc(self):
        """
        the map is occupied by some agents
        this function locates agents once at the beginning!
        Wall is made before the agents, so the agent cannot share the location with the walls
        """
        # Resulting list that will hold the agent positions
        loc = []
        for agt in self.agents:
            a_class = agt.agt_class
            comp = self.wall_loc + (loc[-1] if len(loc) > 0 else loc)
            good = 0

            a_x, a_y = 0, 0
            while good != len(comp):
                if a_class == 2:
                    # Hiders start at the top left part of the map
                    a_x = np.random.randint(low=0, high=int(self.width / 2), size=1)
                    a_y = np.random.randint(low=0, high=int(self.height / 2), size=1)
                else:
                    # Seekers start at the bottom right part of the map
                    a_x = np.random.randint(low=math.floor(self.width / 2), high=self.width, size=1)
                    a_y = np.random.randint(low=math.floor(self.height / 2), high=self.height, size=1)

                for (ox, oy) in comp:
                    if abs(ox - a_x) > agt.size * 2 and abs(oy - a_y) > agt.size * 2:
                        good += 1

            if (a_x, a_y) not in self.wall_loc:
                self.map[(a_x, a_y)] = a_class
                loc.append((a_x, a_y))

        # loc = [tuple(np.array(i).reshape(-1)) for i in loc]
        return loc

    # Get agent states
    def get_agent_state(self, agent_id: int):
        vision = get_vision(self.map, self.agent_loc[agent_id], self.agents[agent_id].angle,
                            self.agents[agent_id].scope)
        sound = get_sound(self.agent_loc, self.agents, agent_id, self.sound_limit)
        comm = get_communication(self.agent_loc, self.agents, agent_id)

        self.agents[agent_id].vision = vision
        self.agents[agent_id].sound = sound

        teammates = [a for i, a in enumerate(self.agents) if i != agent_id and a.agt_class == self.agents[agent_id].agt_class]
        for j, agt in enumerate(teammates):
            agt.comm = comm[j]

        return None

    # World update function
    def update(self):
        actions = []
        for a_i, agt in enumerate(self.agents):
            # Update the dynamic variables of the agent
            agt.vision, agt_sound, agt.comm = self.get_agent_state(a_i)

            actions.append((a_i, agt.action()))

        for act in actions:
            agt_i, (action, action_param) = act

    # TODO: Clean this up
    def refresh_map(self):
        # Empty map 생성하기 (+ borders if requested)
        self.map = np.zeros((self.width - 10, self.height - 10))
        if self.borders:
            self.map = np.pad(self.map, pad_width=5, mode='constant', constant_values=1.)

        for pt in self.wall_loc:
            self.map[pt] = 1

    def init_drawing(self):
        global show_world, envn, world
        self.refresh_map()
        world = self.map
        show_world = cv2.cvtColor((world * 255).astype('uint8'), cv2.COLOR_GRAY2BGR)
        show_world[np.where((show_world == (255, 255, 255)).all(axis=2))] = (255, 0, 0)


if __name__ == "__main__":
    show_world = None
    # ------Agent parameters------
    max_speed = 30
    max_stamina = 10

    acceleration_limits = (15, 90 * math.pi / 180)
    scope = (140, 15)
    size = 5

    envn = World(50, 50, 20, True, 1, 1, max_speed, max_stamina, acceleration_limits, scope, size)
    world = envn.map

    envn.init_drawing()
    for _ in range(10):
        envn.agent_state(show_world)

        cv2.imshow("world", show_world)
        k = cv2.waitKey(0)
        if k == 5:
            break

    """
    self.agent_loc[i][-1][0] += delta_x
    TypeError: 'tuple' object does not support item assignment
    """
