import math
import operator
import random
import sys

import cv2
import numpy as np
from typing import Tuple

# test_agent is in the environment folder
from agent import Agent

# If you get a truncated representation, but want the full array, try
np.set_printoptions(threshold=sys.maxsize)



class Map:

    # 에이 전트 위치
    def __init__(self, max_height: int, max_width: int, max_num_walls: int,
                 borders: bool, num_hiders: int, num_seekers: int,
                 max_speed: float, max_stamina: float, accel_limit: Tuple[float, float],
                 visual_scope: Tuple[int, float], size: int
                 ):

        """
        다양한 환경에서의 학습을 시켜보자
        """
        # alarm near, alarm far -> 30*30 이내에서 나는 "소리" 는 alarm = 5 100*100 이면 alarm= 10 이라고 주고 agent_state에서
        # agent_alarm해주려 했는데, 맵이 이것보다 작게 생성되는 경우가 있어서 바꿔줌 # , alarm_near: int, alarm_far: int

        self.width = np.random.randint(max_width // 2, max_width)
        self.height = np.random.randint(max_height // 2, max_height)
        self.num_walls = np.random.randint(max_num_walls // 2, max_num_walls)
        # 가깝다의 기준? 30?
        a = min(self.width, self.height)
        self.alarm_near = int(a / 20)  # max(1, int(a/20))
        self.alarm_far = int(a / 10)  # max(2, int(a/10))

        self.num_hiders = num_hiders
        self.num_seekers = num_seekers

        print(f'Map Size\nwidth: {self.width}, height: {self.height}\nnum_walls: {self.num_walls}')
        print(f'num_hiders: {self.num_hiders}, num_seekers: {self.num_seekers}')

        # Empty map 생성하기 (+ borders if requested)
        self.map = np.zeros((self.width, self.height))
        self.borders = borders
        if borders:
            self.map = np.pad(self.map, pad_width=5, mode='constant', constant_values=1.)
            self.width += 10
            self.height += 10

        # initial agent and wall locations
        self.wall_loc = self.make_walls()  # 이러면 맵 부르자 마자 wall 함수도 돌아가는

        print("Initial state of the map with agents(1: walls, 2: hiders, 3: seekers)")
        # print(self.map)

        # self.agent_loc = [tuple(np.array(i).reshape(-1)) for i in self.init_agent_loc()]  # list of tuples
        self.agents = []
        self.agent_loc = []
        for i in range(num_hiders):
            self.agents.append(Agent(2, max_speed, max_stamina, accel_limit, visual_scope, size))
            self.agent_loc.append(self.init_agent_loc())
        for j in range(num_seekers):
            self.agents.append(Agent(3, max_speed, max_stamina, accel_limit, visual_scope, size))
            self.agent_loc.append(self.init_agent_loc())

        print(f'self.agent_loc: {self.agent_loc}')
        # print(f'len(self.agent_loc) should be equal to the number of hiders and seekers: {len(self.agent_loc)}')

        # 아래 agent_state 함수 참고
        av = []
        self.agent_vision = [av.append([0]) for i in range(self.num_seekers + self.num_hiders)]

        self.history = [[] for j in range(self.num_seekers + self.num_hiders)]


        # get number of agents nearby 생각해보니까 몇 명이 있는지는 모르고 누군가가 있는것 같다?
        # seeker는 hider에 대한 정보를 어떻게 이용할 것인가? 궁금!
        self.agent_alarm = np.array([[0]] * (self.num_seekers + self.num_hiders))

    def refresh_map(self):
        # Empty map 생성하기 (+ borders if requested)
        self.map = np.zeros((self.width - 10, self.height - 10))
        if self.borders:
            self.map = np.pad(self.map, pad_width=5, mode='constant', constant_values=1.)

        for pt in self.wall_loc:
            self.map[pt] = 1

    # Make walls
    def make_walls(self):  # (x,y) is the starting point
        """
        the map is occupied(1) by some walls
        """
        wall_loc = []

        for i in range(self.num_walls):
            # 벽의 시작점
            x0 = np.random.randint(0, self.width)
            y0 = np.random.randint(0, self.height)

            # length of the wall -> 2부터 시작하는게 맞는 거 같애 not 0! 흠 장애물처럼 1개짜리도 있을 수 있다고 할까 그럼 그냥 0!
            w_len = np.random.randint(0, min(self.width - x0, self.height - y0))

            # shape of the wall (horizontal, vertical, 45) - tuple contains a int
            ang = np.random.choice(3, 1, p=[0.3, 0.3, 0.4])

            if ang[0] == 0:
                x1 = x0 + w_len
                y1 = y0
            elif ang[0] == 1:
                x1 = x0
                y1 = y0 + w_len
            else:
                x1 = x0 + w_len
                y1 = y0 + w_len

            xs = np.arange(x0, x1)
            ys = np.arange(y0, y1)

            # if (x0 == x1) or (y0 == y1) then xs, ys -> []
            if x0 == x1:
                xs = [x0] * w_len
            if y0 == y1:
                ys = [y0] * w_len

            # If the pixel is occupied by the wall, let's put 1 there
            for (x_, y_) in zip(xs, ys):
                self.map[(x_, y_)] = 1
                wall_loc.append((x_, y_))

        # print(f'wall_loc: {wall_loc}')

        return wall_loc

    def init_agent_loc(self):
        """
        the map is occupied by some agents(2)
        this function locates agents once at the beginning!

        Wall is made before the agents, so the agent cannot share the location with the walls
        """
        loc = []

        while len(loc) < self.num_hiders:

            h_x = np.random.randint(low=0, high=int(self.width / 2), size=1)
            h_y = np.random.randint(low=0, high=int(self.height / 2), size=1)

            if (h_x, h_y) not in self.wall_loc:
                self.map[(h_x, h_y)] = 2
                loc.append((h_x, h_y))

        while len(loc) < (self.num_hiders + self.num_seekers):
            s_x = np.random.randint(low=math.floor(self.width / 2), high=self.width, size=1)
            s_y = np.random.randint(low=math.floor(self.height / 2), high=self.height, size=1)

            if (s_x, s_y) not in self.wall_loc:
                self.map[(s_x, s_y)] = 3  # 3 -> 환경은 에이전트 각각이 하이더인지 시커인지 구분할 필요가 있는가?
                loc.append((s_x, s_y))

        # print(f'init_agent_loc is fine.. {len(loc) == (self.num_hiders + self.num_seekers)}')

        loc = [tuple(np.array(i).reshape(-1)) for i in loc]
        return loc

    def agent_state(self, show_world):
        """

        environment should tell agent some information
        self.agent_loc -> 알면 안되지!
        self.agent_vision
        self.agent_alarm

        """
        # for i in range(len(self.agents)):

        for i, agt in enumerate(self.agents):
            x, y = self.agent_loc[i][-1] # latest position of the agent_i
            # world[y, x] = agt.agt_class -> map에서 hider의 위치는 2로, seeker는 3으로 되어있음

            # Draw Agent
            if agt.agt_class == 2:
                color = (200, 150, 0)
            else:
                color = (0, 200, 200)
            cv2.circle(show_world, (x, y), agt.size, color, -1)

            # Get agent vision
            (pa, pb), vis = self.get_vision(world, (x, y), agt.angle, agt.scope)

            # Draw vision
            cv2.imshow("vision", cv2.resize((vis[0] * 255).astype('uint8'), (10, vis[0].shape[0] * 10),
                                            interpolation=cv2.INTER_NEAREST))
            cv2.line(show_world, tuple(int(n) for n in pa), tuple(int(n) for n in pb), (200, 0, 200), 2)

            # Policy
            if 1 in vis[0]:  # If wall in vision, rotate
                action = 1
                action_param = {"ang_accel": (random.randint(20, 45) * math.pi / 180),
                                "accel": -10}

            elif len(set(self.history[i])) == 1:  # If stationary for 3 time steps rotate
                action = 1
                action_param = {"ang_accel": (random.randint(20, 45) * math.pi / 180),
                                "accel": 0}

            elif agt.agt_class == 3 and 2 in vis[0] and vis[1][
                list(vis[0]).index(2)] < 60:  # If hider in front, tag
                action = 2
                action_param = {}

            else:  # When there isn't a special event, just move forward
                action = 1
                action_param = {"ang_accel": (0 * math.pi / 180), "accel": 5}

            # Action based on Policy
            delta_x, delta_y = agt.action(choice=action, **action_param)
            delta_x = min(delta_x, world.shape[1] - (5 + agt.size) - x)
            delta_y = min(delta_y, world.shape[0] - (5 + agt.size) - y)

            if envn is not None:
                # Agent cannot cross walls
                for w in envn.wall_loc:
                    w = w[::-1]
                    # cv2.circle(show_world, tuple(int(ee) for ee in w), 3, (0, 0, 255), -1)
                    if delta_x >= 0 and delta_y >= 0:
                        if x <= w[0] <= x + delta_x and y <= w[1] <= y + delta_y:
                            delta_x, delta_y = 0, 0
                    elif delta_x >= 0 >= delta_y:
                        if x <= w[0] <= x + delta_x and y >= w[1] >= y + delta_y:
                            delta_x, delta_y = 0, 0
                    elif delta_x <= 0 <= delta_y:
                        if x >= w[0] >= x + delta_x and y <= w[1] <= y + delta_y:
                            delta_x, delta_y = 0, 0
                    elif delta_x <= 0 and delta_y <= 0:
                        if x >= w[0] >= x + delta_x and y >= w[1] >= y + delta_y:
                            delta_x, delta_y = 0, 0

            self.agent_loc[i][-1][0] += delta_x
            self.agent_loc[i][-1][1] += delta_y

            # Clamp position to the walls if negative
            self.agent_loc[i][-1][0] = 5 + agt.size if self.agent_loc[i][-1][0] < 5 + agt.size else self.agent_loc[i][-1][0]
            self.agent_loc[i][-1][1] = 5 + agt.size if self.agent_loc[i][-1][1] < 5 + agt.size else self.agent_loc[i][-1][1]

            self.history[i].append((x, y))
            if len(self.history[i]) > 3:
                self.history[i] = self.history[i][1:]


        # 엄청 가까우면(30*30이내) - alarm: 10
        # 근처에 있는거 같으면(100*100이내) - alarm: 5
        # x_axis, y_axis 범위같으니까 하나만 np.arange 해주면 된
        # 범위가 커지면 map 밖으로 벗어날 가능성이 커짐
        alarm_near_x = np.arange(-self.alarm_near, self.alarm_near)
        alarm_far_x = np.arange(-self.alarm_far, self.alarm_far)

        # 지금 내 위치에서 near을 더하면 반경의 포지션이 나오겠지?
        near = [(x, y) for x in alarm_near_x for y in alarm_near_x]
        far = [(x, y) for x in alarm_far_x for y in alarm_far_x]

        for i, agt in enumerate(self.agents):
            x, y = self.agent_loc[i][-1]  # latest position of the agent_i

            als_near = [tuple(map(operator.add, (x, y), n)) for n in near]
            # arange가 크면 맵 밖으로 나가게 되서 clamp an integer(x) to range()
            als_near = [(sorted(al[0], 0, self.width)[1], sorted(al[1], 0, self.height)[1]) for al in als_near]
            self.agent_alarm[i] = 5 if (2 or 3) in [self.map[al] for al in als_near] else 0

            als_far = [tuple(map(operator.add, (x, y), n)) for n in far]
            als_far = [(sorted(al[0], 0, self.width)[1], sorted(al[1], 0, self.height)[1]) for al in als_far]
            self.agent_alarm[i] = 10 if (2 or 3) in [self.map[al] for al in als_far] else 0

        # if agent.stamina == 0: means dead, return None from then, no update anymore

        # return self.agent_vision, self.agent_alarm
        return None

    def init_drawing(self):
        global show_world, envn, world
        self.refresh_map()
        world = self.map
        show_world = cv2.cvtColor((world * 255).astype('uint8'), cv2.COLOR_GRAY2BGR)
        show_world[np.where((show_world == (255, 255, 255)).all(axis=2))] = (255, 0, 0)


    def get_line(self, p1, p2, dist=None):
        x1, y1 = p1
        x2, y2 = p2

        if dist is None:
            dist = int(math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))

        positions = []
        if x1 == x2:
            for yi in np.arange(y1, y2, (y2 - y1) / dist):
                positions.append((x1, yi))
        else:
            grad = (y2 - y1) / (x2 - x1)
            y_intercept = y1 - grad * x1

            for xi in np.arange(x1, x2, (x2 - x1) / dist):
                yi = grad * xi + y_intercept
                positions.append((xi, yi))
        return positions

    def get_vision(self, env, position, orientation, constraints):

        x_pos, y_pos = position
        angular_scope, dist_scope = constraints
        angular_scope = angular_scope * math.pi / 180

        # Distance at left/right limits
        max_dist = dist_scope / math.cos(angular_scope / 2)

        # Get (relative) left limit
        xl = int(x_pos + max_dist * math.cos(orientation - angular_scope / 2))
        yl = int(y_pos + max_dist * math.sin(orientation - angular_scope / 2))

        # right limit
        xr = int(x_pos + max_dist * math.cos(orientation + angular_scope / 2))
        yr = int(y_pos + max_dist * math.sin(orientation + angular_scope / 2))

        # Arrays to hold vision and distance information
        view = []
        dist = []

        # Get visual scope line
        vis_line = self.get_line((xl, yl), (xr, yr), dist=int(2 * math.sqrt(dist_scope ** 2 + max_dist ** 2)))

        # For every line from the agent to the visual limit
        for vis_pt in vis_line:
            vis_pt = (int(vis_pt[0]), int(vis_pt[1]))

            sight_line = self.get_line((x_pos, y_pos), vis_pt)
            sight_value = [env[int(ye), int(xe)] if env.shape[1] > int(xe) > 0 and env.shape[0] > int(ye) > 0 else 1
                           for xe, ye in sight_line]

            # Test for object in sight
            whs_query = [np.inf, np.inf, np.inf]
            if 1 in sight_value[2:]:
                whs_query[0] = sight_value[2:].index(1) + 2
            if 2 in sight_value[2:]:
                whs_query[1] = sight_value[2:].index(2) + 2
            if 3 in sight_value[2:]:
                whs_query[2] = sight_value[2:].index(3) + 2

            # If not object in sight, 0(background) in view at inf dist.
            if whs_query[0] == np.inf and whs_query[1] == np.inf and whs_query[2] == np.inf:
                dist.append(np.inf)
                view.append(0)
            else:
                dist.append(min(whs_query))
                view.append(whs_query.index(min(whs_query)) + 1)

        view = np.asarray(view)
        dist = np.asarray(dist)
        # numba complains, probably due to type differences, just return as tuple for the moment
        # vision = np.stack([view, dist], axis=0)
        return ((xl, yl), (xr, yr)), (view, dist)


# step함수가 클래스 내에 정의될 필요가 있는지, while loop 안에서 타임스텝이 움직일 때마다,
# 에이전트에게 정보를 전달하고, 환경을 업데이트 시켜주고,
#     def step(self, choice, agent_id):
#         self.agent_state(show_world)



if __name__ == "__main__":
    show_world = None
    # ------Agent parameters------
    max_speed = 30
    max_stamina = 10

    acceleration_limits = (15, 90 * math.pi / 180)
    scope = (140, 15)
    size = 5

    envn = Map(50, 50, 20, True, 1, 1, max_speed, max_stamina, acceleration_limits, scope, size)
    world = envn.map

    envn.init_drawing()
    for i in range(10):
        envn.agent_state(show_world)

        cv2.imshow("world", show_world)
        k = cv2.waitKey(0)
        if k == 5:
            break

    """
    self.agent_loc[i][-1][0] += delta_x
    TypeError: 'tuple' object does not support item assignment
    """