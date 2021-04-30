import random
from itertools import permutations
import sys
from skimage.draw import rectangle_perimeter
import cv2
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
SIZE = 1000
NUM_ROOMS = 10

map = np.zeros((SIZE, SIZE))  # np.uint8

TYPE = 1

# ---- POSITIONS OF THE ROOMS ---
# ---- 완전히 랜덤한 ----
# TODO: 좀 더 그리드를 나눠서 하는게 - 맵의 특정 부분에 몰아서 생기는 걸 막을 수도 있고, 방이라는 공간을 겹치지 않게 만들려면
if TYPE == 0:
    coords = list(permutations(range(SIZE), 2))
    room_pos = random.sample(coords, NUM_ROOMS)
    print(f'room_pos: {room_pos}')

# ---- 그리드 이용한 ----
if TYPE == 1:
    coords = list(range(0+int(SIZE / 10), SIZE-int(SIZE / 10), int(SIZE / 10)))
    # SIZE에서 /10으로 그리드로 점이 생긴다고 하면 그것보다 작으면 방이 겹치지는 않겠다!
    coord_x, coord_y = np.meshgrid(coords, coords)
    pos_grid = [(x, y) for (x, y) in zip(coord_x.reshape(-1), coord_y.reshape(-1))]
    room_pos = random.sample(pos_grid, NUM_ROOMS)
    print(f'room_pos: {room_pos}')


# TODO: 교차점은 랜덤으로 방이 될 수도 안될 수도 그럼 그리드로 방이 겹치지 않게는 같이 어떻게 하지 흠
# TODO: height, width로 clip 해줘야함 - np.clip 이용 - 위에서 coords에서부터 바깥쪽이 안되게 했다.

# ---- MAKE ROOMS ----
def make_rooms(room_pos, map):

    xl = room_pos[0] - int(SIZE / 25)
    xr = room_pos[0] + int(SIZE / 25)

    yd = room_pos[1] - int(SIZE / 25)
    yu = room_pos[1] + int(SIZE / 25)

    xs = np.arange(xl, xr+1)
    ys = np.arange(yd, yu+1)

    xrs = [xr] * len(ys)
    xls = [xl] * len(ys)
    yus = [yu] * len(xs)
    yds = [yd] * len(xs)

    for (x1, y1) in zip(xrs, ys):
        map[y1, x1] = 1

    for (x2, y2) in zip(xs, yds):
        map[y2, x2] = 1

    for (x3, y3) in zip(xls, ys):
        map[y3, x3] = 1

    for (x4, y4) in zip(xs, yus):
        map[y4, x4] = 1

    return map

# ---- TEST FOR make_rooms() ----
# sample_map = np.zeros((15,15))
# print(f'sample_map;\n{sample_map}')
# sample_pos = (3,5)
#
# new = make_rooms(sample_pos, sample_map)
# print(f'sample_map with a room:\n{new}')


for i in range(NUM_ROOMS):
    map = make_rooms(room_pos[i], map)

map = cv2.cvtColor((map * 255).astype('uint8'), cv2.COLOR_GRAY2RGB)
map[np.where((map == (255, 255, 255)).all(axis=2))] = (255, 0, 0)

# print(f'map:\n {map}')
cv2.imshow('world', map)
cv2.waitKey(0)
cv2.destroyAllWindows()

for i in range(NUM_ROOMS):
    cv2.circle(map, room_pos[i], 5, (255, 255, 255), -1)
    # world[np.where((world == (255, 255, 255)).all(axis=2))] = (255, 0, 0)

    random_id = random.choice([j for j in range(NUM_ROOMS) if j != i])  # list(range(NUM_AGENTS)).remove(i)

    # 연결 통로 대략적인 느낌의 라인
    cv2.line(map, room_pos[i], room_pos[random_id], (200, 0, 200), 2)



# TODO: 겹치는 거
# world = cv2.cvtColor((map * 255).astype('uint8'), cv2.COLOR_GRAY2BGR)
# world[np.where((world == (255, 255, 255)).all(axis=2))] = (255, 0, 0)

cv2.imshow('SHOW WORLD!', map)
cv2.waitKey(0)
cv2.destroyAllWindows()
