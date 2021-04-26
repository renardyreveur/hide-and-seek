from itertools import product
import math
import sys
import time
from typing import Tuple, Dict

import cv2
import numpy as np
import random
# test_agent is in the environment folder
from agent import Agent, actions
# Get states
from .get_states import get_vision, get_sound, get_communication
from .interesting_walls import *
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
        self.map = np.zeros((self.height, self.width))
        self.borders = borders
        if borders:
            self.map = np.pad(self.map, pad_width=5, mode='constant', constant_values=1.)
            self.width += 10
            self.height += 10

        # Add walls to the map
        # self.wall_loc = self.make_walls()
        self.wall_loc = make_walls(self.num_walls, self.width, self.height, self.map)
        print(f'Map Size:\nWidth: {self.width}, Height: {self.height}\nNum Walls: {self.num_walls}')
        print("\n Map Legend: {1: walls, 2: hiders, 3: seekers}")

        # ---- Agent configurations ----
        num_hiders, num_seekers = agent_config

        # 생성된 에이전트에서 class 가 2인거의 길이, 3인거의 길이 가져와서 해도 되긴 하는데
        self.num_hiders = num_hiders
        self.num_seekers = num_seekers

        self.agents = [Agent(2, **agent_kwargs) if i < num_hiders else Agent(3, **agent_kwargs)
                       for i in range(num_seekers + num_hiders)]
        self.agent_loc = self.init_agent_loc()
        print(f'agent_loc: {self.agent_loc}')
        self.sound_limit = sound_limit

        # List of agents that sent a comm signal
        self.comm_list = []

        # List of agents that sent a tag signal
        self.tag_list = []

        # TODO: zip agents and agent_loc
        print(f'\nAgents:\nHiders: {num_hiders}, Seekers: {num_seekers}')

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
                self.map[ys, xs] = 1
                wall_loc.append(coord)

        return wall_loc


    # Make list that will hold agent locations
    def init_agent_loc(self):
        """
        the map is occupied by some agents
        this function locates agents once at the beginning!
        Wall is made before the agents, so the agent cannot share the location with the walls
        """
        print("\nInitialising Agent Locations...")
        # print("If this takes too long, try generating a different map!")

        # Resulting list that will hold the agent positions
        loc = []

        grids = list(product(range(self.width), range(self.height)))
        print(f'len(grids): {len(grids)}')
        print(f'len(self.wall_loc): {len(self.wall_loc)}')

        # TODO: This loop takes too much time
        start = time.process_time()
        grids = [grid for grid in grids if grid not in self.wall_loc]
        print(f'time: {time.process_time() - start}')

        # Width: 684, Height: 906
        # Num Walls: 19
        # Time:
        print(f'len(grids): {len(grids)}')

        # lower-left possible coords for hiders
        pos_grids_hiders = [(x, y) for (x, y) in grids if x < self.width/2 and y < self.height/2]

        # upper-right possible coords for seekers
        pos_grids_seekers = [(x, y) for (x, y) in grids if x > self.width/2 and y > self.height/2]

        if len(pos_grids_seekers) < self.num_seekers or len(pos_grids_hiders) < self.num_hiders:
            print("Invalid map.. \nTry generating a new map!")

        # TODO: should consider the size of agents and walls

        # Select a starting position for each agent
        for agt in self.agents:
            # Get the agent's class, it's starting pos. depends on it's class
            a_class = agt.agt_class

            if a_class == 2:
                a_x, a_y = random.choice(pos_grids_hiders)
                pos_grids_hiders.remove((a_x, a_y))
            else:
                a_x, a_y = random.choice(pos_grids_seekers)
                pos_grids_seekers.remove((a_x, a_y))

            # Update map with initial location
            self.map[(a_y, a_x)] = a_class
            loc.append((a_x, a_y))

        return loc

    # Get agent states
    def get_agent_state(self, agent_id: int):
        """
        This function will be called at every update state,
        where all the agents get the knowledge of the changed state of the world due to their actions

        3 Main agent states are updated: vision, sound, and communication

        """
        # Get array of what the agent sees in front
        vision = get_vision(self.map, self.agent_loc[agent_id], self.agents[agent_id].angle, self.agents[agent_id].scope)

        # Get estimated sound direction and strength
        sound = get_sound(self.agent_loc, self.agents, agent_id, self.sound_limit)

        self.agents[agent_id].vision = vision
        self.agents[agent_id].sound = sound

        # Get list of agents who sent a communication signal holding their relative bearing and distance
        if agent_id in self.comm_list:
            # Communication is only updated for teammates
            comm = get_communication(self.agent_loc, self.agents, agent_id)
            teammates = [a for i, a in enumerate(self.agents) if
                         i != agent_id and a.agt_class == self.agents[agent_id].agt_class]
            for j, agt in enumerate(teammates):
                agt.comm = comm[j]

        if agent_id in self.tag_list:
            # TODO: Find who this agent tagged, remove them from game if valid
            pass

    # World update function
    def update(self):
        """
        Order: Action -> Env State Change -> Agent State Change -> Reset
        """
        # Refresh map
        self.refresh_map()

        # Update agents
        for a_i, agt in enumerate(self.agents):
            # Get new actions based on the updated agent states
            action, action_param = agt.action()
            acts = {
                1: "move",
                2: "tag",
                3: "communicate"
            }
            print(f"AgentID: {a_i, agt.agt_class}, Action: {acts[action].upper()}, Params: {action_param}")

            # Action results
            dx, dy, tag, comm = getattr(actions, acts[action])(agt, **action_param)

            # Update environment states based on action
            # MOVE (Clamp position to the walls if negative)
            x, y = self.agent_loc[a_i]
            x += min(dx, self.width - (5 + agt.size) - x)
            y += min(dy, self.height - (5 + agt.size) - y)
            self.agent_loc[a_i] = (5 + agt.size if x < 5 + agt.size else x, 5 + agt.size if y < 5 + agt.size else y)
            # Update map with new agent locations
            self.map[self.agent_loc[a_i][::-1]] = agt.agt_class

            # TAG
            if tag:
                self.tag_list.append(a_i)

            # COMM
            if comm:
                self.comm_list.append(a_i)

            # Update the dynamic variables of the agent
            self.get_agent_state(a_i)

        self.comm_list = []
        self.tag_list = []

    # TODO: Clean this up
    def refresh_map(self):
        # Empty map 생성하기 (+ borders if requested)
        self.map = np.zeros((self.height - 10, self.width - 10))
        if self.borders:
            self.map = np.pad(self.map, pad_width=5, mode='constant', constant_values=1.)

        for pt in self.wall_loc:
            self.map[pt[::-1]] = 1

    def init_drawing(self):
        show_world = cv2.cvtColor(self.map.astype('uint8'), cv2.COLOR_GRAY2BGR)
        show_world[np.where((show_world == (1, 1, 1)).all(axis=2))] = (255, 0, 0)
        return show_world
