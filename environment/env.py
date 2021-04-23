import math
import sys
from typing import Tuple, Dict

import cv2
import numpy as np
import random

# test_agent is in the environment folder
from agent import Agent, actions

# Get states
from environment.get_states import get_vision, get_sound, get_communication

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
        print(f'agent_loc: {self.agent_loc}')
        self.sound_limit = sound_limit

        # List of agents that sent a comm signal
        self.comm_list = []

        # List of agents that sent a tag signal
        self.tag_list = []

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

        # 벽의 위치는 not available for agents
        n_avail_posx = [w[0] for w in self.wall_loc]
        n_avail_posy = [w[1] for w in self.wall_loc]

        # 랜덤하게 벽 피하면서, 다른 에이전트도 피하게끔 해야한다.
        while True:
            print("While... ")
            for agt in self.agents:
                # 벽의 두께를 고려해서 벽 안에 에이전트가 놓일 수는 없으니가!
                n_pxs = [px for p in n_avail_posx for px in range(p - 5, p + 5)]  # set(n_avail_posx)
                n_pys = [py for p in n_avail_posy for py in range(p - 5, p + 5)]

                a_class = agt.agt_class

                # 하이더는 bottom right, 인데 벽 근처는 좀 피해서 초이스하게끔
                pxs_h = [px for px in range(int(self.width / 2)) if px not in n_pxs]  # set(n_pxs)
                pys_h = [py for py in range(int(self.height / 2)) if py not in n_pys]

                pxs_s = [px for px in range(int(self.width / 2), self.width) if px not in n_pxs]
                pys_s = [py for py in range(int(self.height / 2), self.height) if py not in n_pys]

                # TODO: position list(pxs_h, pys_h 외 2개) can be empty
                # print(f'len(pxs_h): {len(pxs_h)}, len(pys_h): {len(pys_h)}, len(pxs_s): {len(pxs_s)}, len(pys_s): {len(pys_s)}')
                if 0 in [len(pxs_h), len(pys_h), len(pxs_s), len(pys_s)]:
                    break

                if a_class == 2:
                    # Hiders start at the top left part of the map
                    a_x = random.choice(pxs_h)
                    a_y = random.choice(pys_h)
                    n_pxs.append(a_x)
                    n_pys.append(a_y)
                else:
                    # Seekers start at the bottom right part of the map
                    a_x = random.choice(pxs_s)
                    a_y = random.choice(pys_s)
                    n_pxs.append(a_x)
                    n_pys.append(a_y)

                # comp = self.wall_loc + (loc[-1] if len(loc) > 0 else loc)
                # good = 0
                #
                # a_x, a_y = 0, 0
                # random.choice(r)
                # kk[:int(len(kk)/2)]
                # while good != len(comp):
                #     if a_class == 2:
                #         # Hiders start at the top left part of the map
                #         a_x = random.choice(avail_posx[:int(len(avail_posx) / 2)])
                #         a_y = random.choice(avail_posx[:int(len(avail_posy) / 2)])
                #         # a_x = np.random.randint(low=0, high=int(self.width / 2), size=1)
                #         # a_y = np.random.randint(low=0, high=int(self.height / 2), size=1)
                #     else:
                #         # Seekers start at the bottom right part of the map
                #         a_x = np.random.randint(low=math.floor(self.width / 2), high=self.width, size=1)
                #         a_y = np.random.randint(low=math.floor(self.height / 2), high=self.height, size=1)
                #
                #     for (ox, oy) in comp:
                #         if abs(ox - a_x) > agt.size * 2 and abs(oy - a_y) > agt.size * 2:
                #             good += 1

                # if (a_x, a_y) not in self.wall_loc:
                self.map[(a_x, a_y)] = a_class
                loc.append((a_x, a_y))
            break

        # loc = [tuple(np.array(i).reshape(-1)) for i in loc]
        return loc

    # Get agent states
    def get_agent_state(self, agent_id: int):
        """
        This function will be called at every update state,
        where all the agents get the knowledge of the changed state of the world due to their actions

        3 Main agent states are updated: vision, sound, and communication

        """
        # Get array of what the agent sees in front
        vision = get_vision(self.map, self.agent_loc[agent_id], self.agents[agent_id].angle,
                            self.agents[agent_id].scope)

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
        for a_i, agt in enumerate(self.agents):
            # Get new actions based on the updated agent states
            action, action_param = agt.action()
            # getattr string 써야한대서
            acts = ["move", "tag", "communicate"]
            print(f'action_param: {action_param}')
            if not acts[action-1] == "move":
                dx, dy, tag, comm = getattr(actions, acts[action - 1])(**action_param)
            else:
                dx, dy, tag, comm = getattr(actions, acts[action - 1])(agt, **action_param)


            # Update environment states based on action
            # MOVE (Clamp position to the walls if negative)
            print(f'self.agent_loc in update: {self.agent_loc}')
            x, y = self.agent_loc[a_i]
            x += min(dx, self.width - (5 + agt.size) - x)
            y += min(dy, self.height - (5 + agt.size) - y)
            self.agent_loc[a_i] = (5 + agt.size if x < 5 + agt.size else x, 5 + agt.size if y < 5 + agt.size else y)

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
        self.map = np.zeros((self.width - 10, self.height - 10))
        if self.borders:
            self.map = np.pad(self.map, pad_width=5, mode='constant', constant_values=1.)

        for pt in self.wall_loc:
            self.map[pt] = 1

    def init_drawing(self):
        global show_world, new_world, world
        self.refresh_map()
        new_world = self.map
        show_world = cv2.cvtColor((new_world * 255).astype('uint8'), cv2.COLOR_GRAY2BGR)
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
