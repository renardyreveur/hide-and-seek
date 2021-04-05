import math
import operator
import sys

import numpy as np

from .test_agent import Agent

# If you get a truncated representation, but want the full array, try
np.set_printoptions(threshold=sys.maxsize)

agent = Agent(max_speed=1., max_stamina=1., accel_limit=(1., 1.), visual_scope=(1, 1.), size=2)


class Map:

    # 에이 전트 위치
    def __init__(self, max_height: int, max_width: int, max_num_walls: int,
                 borders: bool = True, num_hiders: int = 1, num_seekers: int = 1):
        """
        다양한 환경에서의 학습을 시켜보자
        """

        self.width = np.random.randint(max_width//2, max_width)
        self.height = np.random.randint(max_height//2, max_height)
        self.num_walls = np.random.randint(1, max_num_walls)

        self.num_hiders = num_hiders
        self.num_seekers = num_seekers

        print(f'Map Size\nwidth: {self.width}, height: {self.height}\nnum_walls: {self.num_walls}')
        print(f'num_hiders: {self.num_hiders}, num_seekers: {self.num_seekers}')

        # Empty map 생성하기 (+ borders if requested)
        self.map = np.zeros((self.width, self.height))
        if borders:
            self.map = np.pad(self.map, pad_width=5, mode='constant', constant_values=1.)
            self.width += 10
            self.height += 10

        # initial agent and wall locations
        self.wall_loc = self.make_walls()  # 이러면 맵 부르자 마자 wall 함수도 돌아가는
        agent_loc = self.init_agent_loc()
        self.agent_loc = [tuple(np.array(i).reshape(-1)) for i in agent_loc]  # list of tuples

        print("Initial state of the map with agents(1: walls, 2: hiders, 3: seekers)")
        # print(self.map)

        # 아래 agent_state 함수 참고
        self.agent_vision = np.array([[0, 0, 0]] * (self.num_seekers + self.num_hiders))

        # get number of agents nearby 생각해보니까 몇 명이 있는지는 모르고 누군가가 있는것 같다?
        # seeker는 hider에 대한 정보를 어떻게 이용할 것인가? 궁금!
        self.agent_alarm = np.array([[0]] * (self.num_seekers + self.num_hiders))

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
                self.map[(s_x, s_y)] = 2  # 3 -> 환경은 에이전트 각각이 하이더인지 시커인지 구분할 필요가 있는가?
                loc.append((s_x, s_y))

        # print(f'init_agent_loc is fine.. {len(loc) == (self.num_hiders + self.num_seekers)}')
        return loc

    def agent_state(self, agent_action: int, agent_id):
        """
        environment should tell agent some information
        self.agent_vision
        self.agent_alarm

        """
        if agent_action == 1:  # move
            self.agent_loc[agent_id] += agent.action(1)

        # self.agent_loc[agent_id]: (x,y) location of the agent_i
        ang = [(0, 1), (1, 1), (1, 0)]
        vis = [tuple(map(operator.add, self.agent_loc[agent_id], a)) for a in ang]

        self.agent_vision[agent_id] = [self.map[v] for v in vis]  # 아 근데 이러면 하이더와 시커를 2, 3으로 나누면 안되는 거군
        # 만약 0번째 에이전트가 눈앞에 세개의 픽셀에 대한 정보를 본다고 할 때 그게, 벽 빈공간 그리고 또다른 에이전트이면
        # agent_vision[0] 은 [1,0,2] 가 되는 것이다

        # 나를 중심으로 3*3 반경에 누군가 있으면 1 없으면 0
        near = [(-1, -1), (-1, 0), (-1, 1), (1, -1), (1, 0), (1, 1), (0, 1), (0, -1)]
        als = [tuple(map(operator.add, self.agent_loc[agent_id], n)) for n in near]
        self.agent_alarm[agent_id] = 1 if 1 in [self.map[al] for al in als] else 0

        # if agent.stamina == 0: means dead, return None from then, no update anymore

        return None


if __name__ == "__main__":
    print("Testing with a single agent")
    test_map = Map(30, 30, 20, 1, 0)
    world = test_map.map
    print(f'Hello World! \n{world}')
    # print(f'type(world): {world.dtype}, {type(world)}')
    # test_map.make_walls()  # If you wanna see the map, turn on the print code!
    h1_init_loc = test_map.agent_loc
    print(f'Initial position: {h1_init_loc}')
