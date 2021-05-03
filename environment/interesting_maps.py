import random
import cv2
import numpy as np
import operator
from scipy.spatial import Voronoi, voronoi_plot_2d

# ---- RANDOM WALK PARAMS ----
SIZE = 700
NUM_WALK = 1000  # SIZE*2
THICKNESS = 14  # SIZE/50
NUM_GENERATING = 7

world = np.ones((SIZE, SIZE))

def get_random_walk_map(world, num_generating, num_walk, thickness):

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

            pos_x = sorted((0, SIZE, pos_x))[1]
            pos_y = sorted((0, SIZE, pos_y))[1]

            space_x = list(range(pos_x - thickness, pos_x + thickness + 1))
            space_x = [max(min(x, SIZE - 1), 0) for x in space_x]
            space_y = list(range(pos_y - thickness, pos_y + thickness + 1))
            space_y = [max(min(y, SIZE - 1), 0) for y in space_y]

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

walls = get_random_walk_map(world, NUM_GENERATING, NUM_WALK, THICKNESS)
# print(f'len(wall_loc): {len(walls)}')


# ---- Map using Voronoi diagram ----
SIZE = 500
world = np.zeros((SIZE, SIZE, 3), np.uint8)
NUM_ROOMS = 10
room_pos = [(np.random.randint(1, SIZE), np.random.randint(1, SIZE)) for _ in range(NUM_ROOMS)]
print(f'room_pos: {room_pos}')


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


# compute Voronoi tesselation
vor = Voronoi(room_pos)

# plot
regions, vertices = voronoi_finite_polygons_2d(vor)

# colorize
for region in regions:
    polygon = vertices[region]
    poly = []
    for a in polygon:
        poly.append([sorted((int(b), 0, SIZE))[1] for b in a])
        x_, y_ = [sorted((int(b), 0, SIZE))[1] for b in a]
        world[y_, x_] = 1

    print(f'polygon: \n{poly}')

    # world = cv2.fillPoly(world, np.array(poly, np.int32), (255, 255, 255))
    cv2.imshow('test', world)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ---- THE SIMPLEST MAP WITH WALLS ----
def make_walls(num_walls, map_width, map_height, map):
    """
    the map is occupied(1) by some walls
    """
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
            map[ys, xs] = 1
            wall_loc.append(coord)

    return wall_loc

# Voronoi algorithm
