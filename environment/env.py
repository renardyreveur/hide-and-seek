import numpy as np


class Map():

    ## 에이 전트 위치
    def __init__(self, height, width, num_walls):
        """

        :param height:
        :param width:
        :param num_walls:
        """

        self.width = width
        self.height = height

        for i in range(num_walls):

        wall_len = min(height, width)

        walls = get_walls()

    # Make walls
    def get_walls(self,x, y):  # (x,y) is the starting point
        """

        :param x:
        :param y:
        :return:
        """
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




# 시작점
x0 = np.random.randint(0,width)
y0 = np.random.randint(0,height)
print(f'x0: {x0}, y0: {y0}')

# 끝나는 점
# x1 = np.random.randint(x0, width)
# y1 = np.random.randint(y0, height)
# 위와 같은 식으로 하면 중간에 occupied 되는 pixel?이 모호? 일단은 정사각형 형태로 되는걸로!

w_len = np.random.randint(0,min(width-x0, height-y0))
print(f'w_len: {w_len}')

cho = np.random.choice(3, 1, p=[0.3, 0.3,0.4])
print(f'choice to make wall: {cho[0]}')

if cho[0] == 0:
  x1 = x0 + w_len
  y1 = y0

elif cho[0] == 1:
  x1 = x0
  y1 = y0 + w_len

elif cho[0] == 2:
  x1 = x0 + w_len
  y1 = y0 + w_len

print(f'(x1,y1): ({x1},{y1})')

# np.where()
xs = np.arange(x0, x1)
if x0 == x1:
  print(type(xs))
  xs = [x0]*w_len

print(f'xs: {xs}')


ys = np.arange(y0, y1)
if not ys:
  print(type(ys))
  ys = [y0]*w_len

print(f'ys: {ys}')



# 연결하면 벽