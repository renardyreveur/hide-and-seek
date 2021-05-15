import operator
import random

import cv2
import numpy as np
from scipy.spatial import Voronoi


def random_walk(world, num_generating):
    num_walk = max(max(world.shape) * 2, 1000)
    thickness = int(max(world.shape) / 50)

    wall_loc = []
    dir_choice = [(1, 0), (-1, 0), (0, 1), (0, -1)]  # RIGHT, LEFT, UP, DOWN

    w, h = world.shape

    for n in range(num_generating):
        pos = (np.random.randint(1, w), np.random.randint(1, h))  # [1, SIZE-1]
        world[pos] = 0
        wall_loc.append(pos)
        print(f'Generating {n + 1}-th trail')
        print(f'pos: {pos}')
        for i in range(num_walk):

            dir = random.choice(dir_choice)
            pos_x, pos_y = tuple(map(operator.add, pos, dir))

            pos_x = sorted((0, w, pos_x))[1]
            pos_y = sorted((0, h, pos_y))[1]

            space_x = list(range(pos_x - thickness, pos_x + thickness + 1))
            space_x = [max(min(x, w - 1), 0) for x in space_x]
            space_y = list(range(pos_y - thickness, pos_y + thickness + 1))
            space_y = [max(min(y, h - 1), 0) for y in space_y]

            space = []
            for x in space_x:
                for y in space_y:
                    world[y, x] = 0
                    space.append((x, y))

            pos = random.choice(space)
        wall_loc.extend(space)

    cv2.imshow('show world!', world)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return wall_loc

# ---- TEST FOR A RANDOM WALK MAP ----
# SIZE = 700
# NUM_WALK = 1000  # SIZE*2
# THICKNESS = 14  # SIZE/50
# NUM_GENERATING = 7

# world = np.ones((SIZE, SIZE))
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


def voronoi(world, num_rooms):
    w, h = world.shape[0], world.shape[1]
    print(f'world.shape: {world.shape}')
    room_pos = [(np.random.randint(1, w), np.random.randint(1, h)) for _ in range(num_rooms)]
    print(f'room_pos: {room_pos}')

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
            x_ = sorted((int(a[0]), 0, w - 1))[1]
            y_ = sorted((int(a[1]), 0, h - 1))[1]
            poly.append((x_, y_))

            world[x_, y_] = 1
            # cv2.circle(world, (y_, x_), 5, (255, 255, 255), -1)

        if len(poly) > 2:
            areas.append(np.array(poly))

        # print(f'areas: {areas}')

        # If you wanna colorize,
        # color = np.random.choice(range(256), size=3)
        # color = (int(color[0]), int(color[1]), int(color[2]))
        # cv2.fillPoly(world, areas, tuple(color))

        # Real map
        room_or_block = ((255, 255, 255), (0, 0, 0))
        color = random.choices(room_or_block, weights=[0.5, 0.5])
        # print(f'color: {color}')
        cv2.fillPoly(world, areas, color[0])

        # 번갈아 가면서 선택하면 조금 더 듬성등성? 될 줄 알았는데
        # color = room_or_block[r]
        # print(f'color; {color}')
        # cv2.fillPoly(world, areas, color)
        # r += 1

    # print(f'world:\n{world}')
    cv2.imshow('test', world)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    world[world == 255] = 1
    w_loc = np.where(world == 1)
    w_loc = [(w_loc[1][i], w_loc[0][i]) for i in range(len(w_loc[0]))]

    return w_loc

# ---- TEST FOR VORONOI ----
# world = np.zeros((500, 500))
# NUM_ROOMS = 50
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

