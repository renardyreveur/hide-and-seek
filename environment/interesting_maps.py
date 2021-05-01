import random
import cv2
import numpy as np
import operator

# ---- RANDOM WALK PARAMS ----
SIZE = 600
MAX_LENGTH = 1000  # SIZE/10
THICKNESS = 10
NUM_GENERATING = 5

world = np.zeros((SIZE, SIZE))
dir_choice = [(1, 0), (-1, 0), (0, 1), (0, -1)]  # RIGHT, LEFT, UP, DOWN

for n in range(NUM_GENERATING):
    pos = (np.random.randint(1, SIZE), np.random.randint(1, SIZE))  # [1, SIZE-1]
    # pos = (int(SIZE/2), int(SIZE/2))  # 확실히 중간에서 시작하는게 랜덤하게 잘 생김 한 번만 제너레이팅 하는경우엔
    world[pos] = 1
    print(f'Generating {n + 1}-th trail')
    print(f'pos: {pos}')
    for i in range(MAX_LENGTH):

        dir = random.choice(dir_choice)
        pos_x, pos_y = tuple(map(operator.add, pos, dir))

        pos_x = sorted((0, SIZE, pos_x))[1]
        pos_y = sorted((0, SIZE, pos_y))[1]

        pos = (pos_x, pos_y)
        # print(f'pos: {pos}')

        # space_x = min(THICKNESS, pos_x, SIZE - pos_x)  너무 경계 근처로 가면 계속 SIZE-pose_x 만 고르게 되서 경계를 벗어나지 못하게 됨

        space_x = list(range(pos_x - THICKNESS, pos_x + THICKNESS + 1))
        space_x = [max(min(x, SIZE - 1), 0) for x in space_x]
        space_y = list(range(pos_y - THICKNESS, pos_y + THICKNESS + 1))
        space_y = [max(min(y, SIZE - 1), 0) for y in space_y]

        space = []
        for x in space_x:
            for y in space_y:
                world[y, x] = 1
                space.append((x, y))

        # 이게 관건이었구만
        pos = random.choice(space)

cv2.imshow('show world!', world)
cv2.waitKey(0)
cv2.destroyAllWindows()




# Voronoi algorithm
