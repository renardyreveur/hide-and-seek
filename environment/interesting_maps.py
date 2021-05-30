import math
import operator
import random
import sys
from random import shuffle
import cv2
import numpy as np
from scipy.spatial import Voronoi

from . import map_util


# If you want to run this script as a main file, comment the above code, and try with the below.
# import map_util


def amongUs(world, num_rooms, border=100):
    world = np.ones(world.shape)

    # print(f'or_world.shape: {world.shape}')
    thickness = int(border / 4)
    # let's add borders in the map - Subtraction of tuples using map, lambda

    nworld_shape = tuple(map(lambda i, j: i - j, world.shape, (border, border)))
    world = np.ones(nworld_shape)
    h, w = world.shape

    dir_choice = [(1, 0), (-1, 0), (0, 1), (0, -1)]  # RIGHT, LEFT, UP, DOWN
    factors = []

    for whole_number in range(1, num_rooms + 1):
        if num_rooms % whole_number == 0:
            factors.append((whole_number, num_rooms // whole_number))

    if len(factors) != 2:  # if num_rooms is a not prime, removing (1, *) to make map more dynamic
        factors = factors[1:-1]

    factor = random.choice(factors)
    num_walks = int(min(world.shape) / 10)

    room_pos = []

    f0 = factor[0]
    f1 = factor[1]

    h_pos = list(range(0, h - thickness, h // f0))
    h_pos.append(h)

    w_pos = list(range(0, w - thickness, w // f1))
    w_pos.append(w)

    # print(f'h_pos: {h_pos}, w_pos: {w_pos}')

    for i in range(len(h_pos) - 1):
        for j in range(len(w_pos) - 1):
            rand_h = np.random.randint(low=h_pos[i], high=h_pos[i + 1])
            rand_w = np.random.randint(low=w_pos[j], high=w_pos[j + 1])
            room_pos.append((rand_w, rand_h))
            # cv2.circle(world, (rand_w, rand_h), 10, 0, -1)

    # cv2.imshow('room_pos', world)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    print(f"\namongUs Map is generating...\nnum_rooms: {len(room_pos)}, num_walks: {num_walks}, thickness: {thickness}")
    print(f'\nroom_pos: {room_pos}')

    # If you want more random paths through rooms, use below
    # random.shuffle(room_pos)
    # room_pos_arr = np.array(room_pos, np.int32)
    # cv2.polylines(world, [room_pos_arr], False, 0, thickness)

    # making paths with minimum spanning tree algorithm
    map_util.mst(room_pos, world, thickness)

    # cv2.imshow('path', world)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # TODO: len(room_pos) can not be equal to num_rooms; I let it to be different sometimes, just to make more rooms.

    for n in range(len(room_pos)):

        print(f'Generating a trail from the {n}-th room...')

        pos = (room_pos[n][1], room_pos[n][0])
        # print(f'pos: {pos}')

        world[pos] = 0

        for i in range(num_walks):

            dir = random.choice(dir_choice)
            pos_x, pos_y = tuple(map(operator.add, pos, dir))  # (h, w)

            pos_x = sorted((0, h, pos_x))[1]
            pos_y = sorted((0, w, pos_y))[1]

            space_x = list(range(pos_x - thickness, pos_x + thickness + 1))
            space_x = [sorted((0, h - 1, x))[1] for x in space_x]
            space_y = list(range(pos_y - thickness, pos_y + thickness + 1))
            space_y = [sorted((0, w - 1, y))[1] for y in space_y]

            space = []
            for x in space_x:
                for y in space_y:
                    world[x, y] = 0
                    space.append((x, y))

            pos = random.choice(space)
            # print(f'space: {space}\npos: {pos}')

        # cv2.imshow('show world!', world)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    world = np.pad(world, pad_width=int(border / 2), mode='constant', constant_values=0.)

    wall_loc = np.where(world == 1)
    wall_loc = [(wall_loc[1][i], wall_loc[0][i]) for i in range(len(wall_loc[0]))]

    return wall_loc


# ---- TEST FOR AMONGUS MAP ----
# import sys
# np.set_printoptions(threshold= sys.maxsize)
# world = np.zeros((600, 800))
# num_rooms = 50
# amongUs(world, num_rooms)


def random_walk(world, num_walls):
    # or_w, or_h = world.shape
    # Make a larger map and then resize it to make more complex map! -> I think it's not good for a random_walk map
    # world = cv2.resize(world, (or_w * 2, or_h * 2))  # *10

    world = np.ones(world.shape)
    num_walk = int(min(world.shape) / num_walls)
    thickness = 10

    print(
        f"\nRandom Walk Map is generating... \nnum_walk: {num_walk}, thickness: {thickness}, num_generating: {num_walls}")

    dir_choice = [(1, 0), (-1, 0), (0, 1), (0, -1)]  # RIGHT, LEFT, UP, DOWN

    h, w = world.shape

    for n in range(num_walls * 2):
        pos = (np.random.randint(1, h), np.random.randint(1, w))  # [1, SIZE-1]
        world[pos] = 0

        print(f'Generating {n + 1}-th trail')
        # print(f'pos: {pos}')

        for i in range(num_walk):

            dir = random.choice(dir_choice)
            pos_x, pos_y = tuple(map(operator.add, pos, dir))

            pos_x = sorted((0, h, pos_x))[1]
            pos_y = sorted((0, w, pos_y))[1]

            space_x = list(range(pos_x - thickness, pos_x + thickness + 1))
            space_x = [max(min(x, h - 1), 0) for x in space_x]
            space_y = list(range(pos_y - thickness, pos_y + thickness + 1))
            space_y = [max(min(y, w - 1), 0) for y in space_y]

            space = []
            for x in space_x:
                for y in space_y:
                    world[x, y] = 0
                    space.append((x, y))

            pos = random.choice(space)

    # cv2.imshow('show world!', world)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Resize the map back to the original size
    # world = cv2.resize(world, (or_h, or_w))

    wall_loc = np.where(world == 0)
    wall_loc = [(wall_loc[1][i], wall_loc[0][i]) for i in range(len(wall_loc[0]))]

    return wall_loc


# ---- TEST FOR A RANDOM WALK MAP ----
# SIZE = 500
# NUM_GENERATING = 7
#
# world = np.ones((SIZE, SIZE*2))
# walls = random_walk(world, NUM_GENERATING)
# print(f'len(wall_loc): {len(walls)}')


# ---- Map using Voronoi diagram ----

def voronoi(world, num_rooms, enl_w=5):
    num_rooms *= enl_w
    or_w, or_h = world.shape

    # Make a larger map and then resize it to make more sophisticated map!
    world = cv2.resize(world, (or_w * enl_w, or_h * enl_w))  # *10

    w, h = world.shape
    room_pos = [(np.random.randint(1, h), np.random.randint(1, w)) for _ in range(num_rooms * enl_w)]  # *10
    # print(f'room_pos: {room_pos}')

    # compute Voronoi tesselation
    vor = Voronoi(room_pos)

    # plot
    regions, vertices = map_util.voronoi_finite_polygons_2d(vor)

    # 번갈아가면서 하려면
    # room_or_block = ((255, 255, 255), (0, 0, 0))*10
    # r = 0
    # real map 사용하려면 아래에 real map 부분 켜줘야함
    for region in regions:
        polygon = vertices[region]
        areas = []
        poly = []
        for a in polygon:
            x_ = sorted((int(a[0]), 0, h - 1))[1]
            y_ = sorted((int(a[1]), 0, w - 1))[1]
            poly.append((x_, y_))

            world[y_, x_] = 1
            # cv2.circle(world, (y_, x_), 5, (255, 255, 255), -1)

        if len(poly) > 2:
            areas.append(np.array(poly))

        # print(f'areas: {areas}')

        # If you wanna colorize, and if the len(world.shape) == 3 not 2
        # color = np.random.choice(range(256), size=3)
        # color = (int(color[0]), int(color[1]), int(color[2]))
        # cv2.fillPoly(world, areas, tuple(color))

        # Real map
        room_or_block = ((255, 255, 255), (0, 0, 0))
        color = random.choices(room_or_block, weights=[0.2, 0.8])
        # print(f'color: {color}')
        cv2.fillPoly(world, areas, color[0])

        # 번갈아 가면서 선택하면 조금 더 듬성등성? 될 줄 알았는데
        # color = room_or_block[r]
        # print(f'color; {color}')
        # cv2.fillPoly(world, areas, color)
        # r += 1

    # If you wanna see the map before resized (or check whether the code above is working or not)
    # cv2.imshow('test', world)
    # cv2.waitKey(0)

    world = cv2.resize(world, (or_h, or_w))

    # cv2.imshow('test', world)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    world[world == 255] = 1
    w_loc = np.where(world == 1)
    w_loc = [(w_loc[1][i], w_loc[0][i]) for i in range(len(w_loc[0]))]

    return w_loc


# ---- TEST FOR VORONOI ----
# world = np.zeros((500, 300))
# NUM_ROOMS = 200
# voronoi(world, NUM_ROOMS)


# ---- THE SIMPLEST MAP WITH WALLS ----
def slimWall(world, num_walls):
    """
    the map is occupied(1) by some walls
    """
    map_width, map_height = world.shape
    wall_loc = []
    for i in range(num_walls):
        # Starting point of the wall
        x0 = np.random.randint(0, map_width)
        y0 = np.random.randint(0, map_height)

        # length of the wall
        w_len = np.random.randint(0, min(map_width - x0, map_height - y0))

        # shape of the wall (horizontal, vertical, 45) - tuple contains a int
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
            world[coord] = 1
            wall_loc.append((coord[1], coord[0]))

        cv2.imshow('world', world)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return wall_loc


# ---- TEST FOR WALL TYPE MAP ----
# wor = np.zeros((500, 300))
# slimWall(wor, 20)


def fatWall(world, num_walls, thickness=5):
    # world = np.ones(world.shape)
    h, w = world.shape

    factors = []

    for whole_number in range(1, num_walls + 1):
        if num_walls % whole_number == 0:
            factors.append((whole_number, num_walls // whole_number))

    if len(factors) != 2:  # if num_rooms is a not prime, removing (1, *) to make map more dynamic
        factors = factors[1:-1]

    factor = random.choice(factors)

    f0 = factor[0]
    f1 = factor[1]

    h_pos = list(range(0, h - thickness, h // f0))
    h_pos.append(h)

    w_pos = list(range(0, w - thickness, w // f1))
    w_pos.append(w)

    # print(f'h_pos: {h_pos}, w_pos: {w_pos}')

    wall_pos = []

    for i in range(len(h_pos) - 1):
        for j in range(len(w_pos) - 1):
            rand_h = np.random.randint(low=h_pos[i], high=h_pos[i + 1])
            rand_w = np.random.randint(low=w_pos[j], high=w_pos[j + 1])
            wall_pos.append((rand_w, rand_h))
            # cv2.circle(world, (rand_w, rand_h), 10, 0, -1)

    wall_pos.sort(key=lambda x: x[0])

    for i in range(math.floor(len(wall_pos) / 2)):
        # rand_pts = random.sample(wall_pos, 2)
        # y0, x0 = rand_pts[0]
        # y1, x1 = rand_pts[1]
        # wall_pos = [i for i in wall_pos if i not in rand_pts]

        y0, x0 = wall_pos[i * 2]
        y1, x1 = wall_pos[i * 2 + 1]

        # Get line
        cv2.line(world, (y0, x0), (y1, x1), 255, thickness)

    # cv2.imshow('world', world)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # print(f'world:\n{world}')
    world[world == 255] = 1  # necessary?
    w_loc = np.where(world == 1)
    w_loc = [(w_loc[1][i], w_loc[0][i]) for i in range(len(w_loc[0]))]

    return w_loc


# ---- TEST FOR FAT WALL TYPE ----
# wor = np.zeros((600, 900))
# fatWall(wor, 20)


# Binary tree maze map
def binaryMaze(world, num_paths, border=50):
    world = np.ones(world.shape)

    # print(f'or_world.shape: {world.shape}')
    thickness = int(border / 4)
    # let's add borders in the map - Subtraction of tuples using map, lambda

    nworld_shape = tuple(map(lambda i, j: i - j, world.shape, (border, border)))
    print(f'nworld_shape: {nworld_shape}')
    world = np.ones(nworld_shape)
    h, w = world.shape

    dir_choice = [(1, 0), (0, 1), (-1, 0), (0, -1)]  # Binary way
    # max_len = int(min(world.shape)/50)
    max_len = thickness * 3
    print(f'max_len: {max_len}')

    start_ps = [(1, 0), (0, 1)]
    start_ps = start_ps * (min(world.shape) // 100)

    s_max = list(range(min(world.shape) // 100))
    multip = list(np.repeat(s_max, 2))

    print(f'start_ps: {start_ps}, multip: {multip}')
    start_ps = [(start_ps[i][0]*100*multip[i], start_ps[i][1]*100*multip[i]) for i in range(len(multip))]
    start_ps = list(set(start_ps))
    print(f'start_ps: {start_ps}')

    for i in range(len(start_ps)):
        h0, w0 = start_ps[i]
        # h0, w0 = 0, 0
        # h0, w0 = random.randint(0, int(world.shape[0]/2)), random.randint(0, int(world.shape[1]/2))
        while h0 < world.shape[0] or w0 < world.shape[1]:
            try:
                wall_len = random.randint(thickness, max_len)
                dir_id = np.random.choice(list(range(4)), 1, p=[0.3, 0.3, 0.2, 0.2])[0]
                wall_dir = dir_choice[dir_id]
                wall_dir = tuple([i * wall_len for i in wall_dir])

                h1, w1 = tuple(map(lambda i, j: i + j, (h0, w0), wall_dir))
                cv2.line(world, (h0, w0), (h1, w1), 0, 30)
                # cv2.imshow('world', world)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                h0, w0 = h1, w1
            except Exception as e:
                print(e)

        # cv2.imshow('world', world)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    world = np.pad(world, pad_width=int(border / 2), mode='constant', constant_values=0.)

    wall_loc = np.where(world == 1)
    wall_loc = [(wall_loc[1][i], wall_loc[0][i]) for i in range(len(wall_loc[0]))]

    # cv2.imshow('world', world)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return wall_loc


# ---- TEST for Binary Maze Map ----
# wor = np.zeros((600, 800))
# binaryMaze(wor, 0)
