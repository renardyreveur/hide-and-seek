# Making more interesting map with using various types of walls

# Map with dangerous beasts
def make_walls_jurassic():
    pass

# Map with many rooms
def make_walls_room():
    pass

# Maze Map
def make_walls_maze():
    pass

import numpy as np

# The simplest walls
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
