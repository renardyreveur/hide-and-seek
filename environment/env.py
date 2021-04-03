import numpy as np


#  If you get a truncated representation, but want the full array, try this!
# np.set_printoptions(threshold=sys.maxsize)

class Map:
    # 에이 전트 위치
    def __init__(self, max_height, max_width, max_num_walls):
        """
        init 에선 random한 크기의 empty한 맵 설정 및 파라미터 설정
        random하게 많이 생성하여 다양한 환경에서의 학습을 시켜본다.
        :param max_height:
        :param max_width:
        :param max_num_walls:
        """

        self.width = np.random.randint(max_width//2, max_width)
        self.height = np.random.randint(max_height//2, max_height)
        self.num_walls = np.random.randint(1, max_num_walls)

        # Empty map 생성하기
        self.map = np.zeros((self.width, self.height))

    # Make walls
    def make_walls(self):  # (x,y) is the starting point
        """
        :return: map occupied(1) by some walls
        """

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

            # If the pixel? is occupied by the wall, let's put 1 there
            for (x_, y_) in zip(xs, ys):
                self.map[(x_, y_)] = 1

        # print(self.map)

    def env_state(self):
        """
        agent가 차지하고 있는 위치
        wall이 차지하고 있는 위
        :return:
        """

    def agent_state(self):
        """
        각 agent의 위치
        살아있는 hiders and seekers의 수?
        :return:
        """


if __name__ == "__main__":
    test_map = Map(20, 20, 10)
    test_map.make_walls()  # If you wanna see the map, turn on the print code!
