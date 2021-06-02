import math
import random
import sys
from itertools import product
from typing import Tuple, Dict

import cv2
import numpy as np

# test_agent is in the environment folder
from agent import Agent, actions
# Get states
from .get_states import get_vision, get_sound, get_communication

# from .interesting_maps import *
from . import interesting_maps

# If you get a truncated representation, but want the full array, try
np.set_printoptions(threshold=sys.maxsize)


class World:
    def __init__(self, map_type: str, agent_config: Tuple[int, int], agent_kwargs: Dict,
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
        # self.num_walls = 20

        print(f'Map Size:\nWidth: {self.width}, Height: {self.height}\nNum Walls: {self.num_walls}')
        print("\n Map Legend: {1: walls, 2: hiders, 3: seekers}")

        # Create empty map (+ borders if requested)
        self.map = np.zeros((self.height, self.width))
        self.borders = borders
        if borders:
            self.map = np.pad(self.map, pad_width=5, mode='constant', constant_values=1.)
            self.width += 10
            self.height += 10

        print(f'Map Size with borders: {self.map.shape}')
        # Add walls to the map
        self.wall_loc = getattr(interesting_maps, map_type)(self.map, self.num_walls)
        # print(f'wall_loc: {self.wall_loc}')

        # ---- Agent configurations ----
        num_hiders, num_seekers = agent_config

        # 생성된 에이전트에서 class 가 2인거의 길이, 3인거의 길이 가져와서 해도 되긴 하는데
        self.num_hiders = num_hiders
        self.num_seekers = num_seekers

        self.agents = [Agent(i, 2, **agent_kwargs) if i < num_hiders else Agent(i, 3, **agent_kwargs)
                       for i in range(num_seekers + num_hiders)]
        self.agent_loc = self.init_agent_loc()
        print(f'agent_loc: {self.agent_loc}')
        self.sound_limit = sound_limit

        # List of agents that sent a comm signal
        self.comm_list = []

        # List of agents that sent a tag signal
        self.tagger_list = []

        # TODO: zip agents and agent_loc
        print(f'\nAgents:\nHiders: {num_hiders}, Seekers: {num_seekers}')

    # Make list that will hold agent locations
    def init_agent_loc(self):
        """
        the map is occupied by some agents
        this function locates agents once at the beginning!
        Wall is made before the agents, so the agent cannot share the location with the walls
        """
        print("\nInitialising Agent Locations...")

        # Resulting list that will hold the agent positions
        loc = {}

        # Get Cartesian product of the map to get all the possible coordinates
        grids = list(product(range(self.width), range(self.height)))
        # Remove wall locations from the possible coordinates
        grids = list(set(grids) - set(self.wall_loc))

        # lower-left possible coords for hiders
        pos_grids_hiders = [(x, y) for (x, y) in grids if x < self.width / 2 and y < self.height / 2]
        # upper-right possible coords for seekers
        pos_grids_seekers = [(x, y) for (x, y) in grids if x > self.width / 2 and y > self.height / 2]

        # If you can't choose enough coordinates for all the agents (unlikely, but possible)
        if len(pos_grids_seekers) < self.num_seekers or len(pos_grids_hiders) < self.num_hiders:
            print("Invalid map.. \nTry generating a new map!")
            raise RuntimeError

        # TODO: The walls can be within the agent due to the agent's size, fix?
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
            loc.update({agt.uid: (a_x, a_y)})

        return loc

    # Get agent states
    def get_agent_state(self, agent_id: int):
        """
        This function will be called at every update state,
        where all the agents get the knowledge of the changed state of the world due to their actions

        3 Main agent states are updated: vision, sound, and communication
        """
        agent = next((x for x in self.agents if x.uid == agent_id), None)

        # Tag event
        if agent_id in self.tagger_list:
            # Get the vision information of the agent in question
            tag_vis = agent.vision
            if 2 in tag_vis[1][0]:  # Check seeker in vision
                tag_vis_idx = list(tag_vis[1][0]).index(2)
                tag_dist_idx = int(tag_vis[1][1][tag_vis_idx])
                x1, y1 = tag_vis[2][tag_vis_idx][tag_dist_idx]  # Get approx location of tagged

                # TODO : TAG condition is still a bit iffy
                # For all the other agents, test if tag was possible (Compare approx loc. with real loc.)
                for a_id, (x2, y2) in self.agent_loc.items():
                    aiq = next((x for x in self.agents if x.uid == a_id), None)
                    if int(math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)) < aiq.size * 2 and aiq.agt_class == 2:
                        self.agents.remove(aiq)
                        self.agent_loc.pop(a_id)
                        return 2
                        # TODO : Reward seeker

        # Get array of what the agent sees in front
        vision = get_vision(self.map, self.agent_loc[agent_id], agent.angle, agent.scope)

        # Get estimated sound direction and strength
        sound = get_sound(self.agent_loc, self.agents, agent_id, self.sound_limit)

        # Update agent's state
        agent.vision, agent.sound = vision, sound

        # Get list of agents who sent a communication signal holding their relative bearing and distance
        if agent_id in self.comm_list:
            # Communication is only updated for teammates
            comm = get_communication(self.agent_loc, self.agents, agent_id)
            teammates = [a for a in self.agents if a.uid != agent_id and a.agt_class == agent.agt_class]
            for agt in teammates:
                agt.comm = comm[agt.uid]

    # World update function
    def update(self):
        """
        Order: Action -> Env State Change -> Agent State Change -> Reset
        """
        # Refresh map
        self.refresh_map()

        # Update agents
        for agt in self.agents:
            # Get new actions based on the updated agent states
            action, action_param = agt.action()
            acts = {
                1: "move",
                2: "tag",
                3: "communicate"
            }
            print(f"AgentID: {agt.uid, agt.agt_class}, Action: {acts[action].upper()}, Params: {action_param}")

            # Action results
            dx, dy, tag, comm = getattr(actions, acts[action])(agt, **action_param)

            # Update environment states based on action
            # MOVE (Clamp position to the walls if negative)
            x, y = self.agent_loc[agt.uid]
            # Agent cannot cross walls
            for w in self.wall_loc:
                if dx >= 0 and dy >= 0:
                    if x <= w[0] <= x + dx and y <= w[1] <= y + dy:
                        dx, dy = 0, 0
                elif dx >= 0 >= dy:
                    if x <= w[0] <= x + dx and y >= w[1] >= y + dy:
                        dx, dy = 0, 0
                elif dx <= 0 <= dy:
                    if x >= w[0] >= x + dx and y <= w[1] <= y + dy:
                        dx, dy = 0, 0
                elif dx <= 0 and dy <= 0:
                    if x >= w[0] >= x + dx and y >= w[1] >= y + dy:
                        dx, dy = 0, 0

            x += min(dx, self.width - (5 + agt.size) - x)
            y += min(dy, self.height - (5 + agt.size) - y)
            self.agent_loc[agt.uid] = (5 + agt.size if x < 5 + agt.size else x, 5 + agt.size if y < 5 + agt.size else y)

            # Update map with new agent locations
            self.map[self.agent_loc[agt.uid][::-1]] = agt.agt_class

            # TAG
            if tag:
                self.tagger_list.append(agt.uid)

            # COMM
            if comm:
                self.comm_list.append(agt.uid)

            # Update the dynamic variables of the agent
            self.get_agent_state(agt.uid)

        self.comm_list = []
        self.tagger_list = []

    def refresh_map(self):
        self.map = np.zeros((self.height - 10, self.width - 10))
        if self.borders:
            self.map = np.pad(self.map, pad_width=5, mode='constant', constant_values=1.)

        print(f'self.map.shape: {self.map.shape}')
        for pt in self.wall_loc:
            self.map[pt[::-1]] = 1  # IndexError

    def init_drawing(self):

        show_world = cv2.cvtColor(self.map.astype('uint8'), cv2.COLOR_GRAY2BGR)
        show_world[np.where((show_world == (1, 1, 1)).all(axis=2))] = (255, 0, 0)
        return show_world
