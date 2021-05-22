import operator
import random

import cv2
import numpy as np
from scipy.spatial import Voronoi
from . import map_util
# 이 파일을 메인으로 돌릴 때는 하고 아래로 고쳐야함.
# import map_util

# TODO: Since the map depends on too much randomness, sometimes it generates the invalid map, making the blocked area
#  where the agents unable to get out of that space, I tried to solve this problem with generating random walk
#  repeatedly, however, it can't be the general solution for this problem. So I'll try to make amongUs_like map,
#  literally the map with rooms

# TODO: num_rooms can not be a prime number

def amongUs(world, num_rooms, border=80):

    oh, ow = world.shape
    print(f'or_world.shape: {world.shape}')
    thickness = int(border/4)
    # let's add borders in the map - Subtraction of tuples using map, lambda

    nworld_shape = tuple(map(lambda i, j: i - j, world.shape, (border, border)))
    world = np.ones(nworld_shape)
    # thickness = int(max(world.shape) / 100)
    # world = np.ones(world.shape)

    dir_choice = [(1, 0), (-1, 0), (0, 1), (0, -1)]  # RIGHT, LEFT, UP, DOWN
    print(f'nworld.shape: {world.shape}')
    h, w = world.shape
    # TODO: making grids of the map first, grids for locating each room.
    factors = []

    for whole_number in range(1, num_rooms + 1):
        if num_rooms % whole_number == 0:
            factors.append((whole_number, num_rooms//whole_number))

    factors = factors[1:-1]
    factor = random.choice(factors)
    print(f'factor: {factor}')
    # num_walks = int(min(world.shape)/min(factor)) * int(max(world.shape)/max(factor))

    # num_walks = min(world.shape)
    # num_walks = 50
    # print(f"\namongUs Map is generating...\nnum_rooms: {num_rooms}, num_walks: {num_walks}")

    room_pos = []

    a = [0, 1]
    ran_facid = random.choice(a)
    # f0 = random.choice(factor)  # number of rooms in a col
    # f1 = [_ for _ in factor if _ != f0][0]  # number of rooms in a row - error if (3, 3)
    f0 = factor[ran_facid]
    a.remove(ran_facid)

    f1 = factor[a[0]]

    num_walks = 50
    # num_walks = int(max(world.shape)//num_rooms)
    print(f'ideal num_walks: {h // f0}, {w//f1} or {num_walks}')

    h_pos = list(range(0, h, h//f0))
    h_pos.append(oh)

    w_pos = list(range(0, w, w//f1))
    w_pos.append(ow)

    print(f'h_pos: {h_pos}\nw_pos: {w_pos}')

    for i in range(len(h_pos)-1):
        for j in range(len(w_pos)-1):
            # if (i != len(h_pos) - 1) and (j != len(w_pos) - 1):
            rand_h = np.random.randint(low=h_pos[i], high=h_pos[i+1])
            rand_w = np.random.randint(low=w_pos[j], high=w_pos[j+1])
            # room_pos.append((rand_w, rand_h))
            room_pos.append((rand_w, rand_h))
            # else:
                # rand_h0 = np.random.randint(h_pos[-1]-thickness, h)
                # rand_h1 = np.random.randint(0, h)
                # rand_w0 = np.random.randint(w_pos[-1]-thickness, w)
                # rand_w1 = np.random.randint(0, w)

                # room_pos.append((random.choice((rand_w0, rand_w1)), random.choice((rand_h0, rand_h1))))
                # room_pos.append((random.choice((rand_h0, rand_h1)), random.choice((rand_w0, rand_w1))))

            # except Exception as e:
            #     print(e)

    print(f'len(room_pos): {len(room_pos)} num_rooms: {num_rooms}')
    # print(f'room_pos: {room_pos}')

    # random.shuffle(room_pos)

    # Pythonic way?
    ran_room_pos = random.choices(room_pos, k=len(room_pos))
    # i = 0
    # for point1, point2 in zip(room_pos, room_pos[1:]):
    #     print(f'point1: {point1}, point2: {point2}')  # should be a tuple
    #     cv2.line(world, point1, point2, 0, thickness)

    # room_pos_arr = np.array(room_pos, np.int32)
    # cv2.drawContours(world, [room_pos_arr], -1, 0, int(thickness*1.5))
    map_util.mst(room_pos, world)
    # cv2.imshow('path', world)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # cv2.polylines(world, [room_pos_arr], False, 0, thickness)

    # pass

    # TODO: assert len(room_pos) == num_rooms, "len(room_pos) != num_rooms..."
    # assert len(room_pos) == num_rooms, "len(room_pos) != num_rooms..."

    # world = np.ones((oh, ow))
    world = np.pad(world, pad_width=int(border/2), mode='constant', constant_values=0.)

    for n in range(len(room_pos)):
        pos = (room_pos[n][1], room_pos[n][0])
        world[pos] = 0
        # wall_loc.append(pos)
        print(f'Generating {n + 1}-th trail')
        print(f'pos: {pos}')
        for i in range(num_walks):

            dir = random.choice(dir_choice)
            pos_x, pos_y = tuple(map(operator.add, pos, dir))

            pos_x = sorted((0, oh, pos_x))[1]
            pos_y = sorted((0, ow, pos_y))[1]

            space_x = list(range(pos_x - thickness, pos_x + thickness + 1))
            space_x = [max(min(x, oh - 1), 0) for x in space_x]
            space_y = list(range(pos_y - thickness, pos_y + thickness + 1))
            space_y = [max(min(y, ow - 1), 0) for y in space_y]

            space = []
            for x in space_x:
                for y in space_y:
                    world[x, y] = 0
                    space.append((x, y))

            pos = random.choice(space)
        # wall_loc.extend(space)
        print("Generating Done!")

    # cv2.imshow('show world!', world)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # print(f'world: \n{world}')
    wall_loc = np.where(world == 1)
    wall_loc = [(wall_loc[1][i], wall_loc[0][i]) for i in range(len(wall_loc[0]))]

    return wall_loc

# ---- TEST FOR AMONGUS MAP ----
# import sys
# np.set_printoptions(threshold= sys.maxsize)
# world = np.zeros((600, 900))
# num_rooms = 20
# amongUs(world, num_rooms)


def random_walk(world, num_walls):
    # or_w, or_h = world.shape
    # Make a larger map and then resize it to make more sophisticated map!
    # world = cv2.resize(world, (or_w * 2, or_h * 2))  # *10

    world = np.ones(world.shape)
    # num_walk = min(min(world.shape) * 2, 1000)
    num_walk = max(world.shape)
    thickness = int(max(world.shape) / 100)
    print(
        f"\nRandom Walk Map is generating... \nnum_walk: {num_walk}, thickness: {thickness}, num_generating: {num_walls}")

    # wall_loc = []
    dir_choice = [(1, 0), (-1, 0), (0, 1), (0, -1)]  # RIGHT, LEFT, UP, DOWN

    h, w = world.shape

    for n in range(num_walls):
        pos = (np.random.randint(1, h), np.random.randint(1, w))  # [1, SIZE-1]
        world[pos] = 0
        # wall_loc.append(pos)
        print(f'Generating {n + 1}-th trail')
        print(f'pos: {pos}')
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
        # wall_loc.extend(space)
        # print("Generating Done!")

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

# https://gist.github.com/pv/8036995
def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.

    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.

    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.

    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1]  # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)


def voronoi(world, num_rooms, enl_w=5):
    or_w, or_h = world.shape

    # Make a larger map and then resize it to make more sophisticated map!
    world = cv2.resize(world, (or_w * enl_w, or_h * enl_w))  # *10

    w, h = world.shape
    room_pos = [(np.random.randint(1, h), np.random.randint(1, w)) for _ in range(num_rooms * enl_w)]  # *10
    # print(f'room_pos: {room_pos}')

    # compute Voronoi tesselation
    vor = Voronoi(room_pos)

    # plot
    regions, vertices = voronoi_finite_polygons_2d(vor)

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
        color = random.choices(room_or_block, weights=[0.4, 0.6])
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
    # print(f'world:\n{world}')
    # cv2.imshow('test', world)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    print(f'world.shape (int_maps): {world.shape}')
    world[world == 255] = 1
    w_loc = np.where(world == 1)
    w_loc = [(w_loc[1][i], w_loc[0][i]) for i in range(len(w_loc[0]))]

    return w_loc


# ---- TEST FOR VORONOI ----
# world = np.zeros((500, 300))
# NUM_ROOMS = 200
# voronoi(world, NUM_ROOMS)


# ---- THE SIMPLEST MAP WITH WALLS ----
def walls(map, num_walls):
    """
    the map is occupied(1) by some walls
    """
    map_width, map_height = map.shape
    wall_loc = []
    for i in range(num_walls):
        # Starting point of the wall
        x0 = np.random.randint(0, map_width)
        y0 = np.random.randint(0, map_height)

        # length of the wall
        w_len = np.random.randint(0, min(map_width - x0, map_height - y0))

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
            map[xs, ys] = 1
            wall_loc.append(coord)

    return wall_loc
