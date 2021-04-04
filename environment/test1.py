# Test with a single agent (one hider)
import math
import random

import cv2
import numpy as np
from numba import njit, jit

from test_agent import Agent
from env import Map

# ---- ENVIRONMENT ----
BASE = False
SIZE = 1000

show_world = None
if BASE:
    world = np.zeros((SIZE, SIZE))
    # Basic walls
    world[0:5, :] = 1
    world[SIZE - 6: SIZE - 1, :] = 1
    world[:, 0:5] = 1
    world[:, SIZE - 6: SIZE - 1] = 1
else:
    print("Testing environment")
    envn = Map(SIZE, SIZE, 10, 1, 0)
    # envn.make_walls()
    world = envn.map
    # world[0:5, :] = 1
    # world[world.shape[0] - 6: world.shape[0] - 1, :] = 1
    # world[:, 0:5] = 1
    # world[:, world.shape[1] - 6: world.shape[1] - 1] = 1

# ---- SINGLE AGENT ----
max_speed = 30
max_stamina = 10
acceleration_limits = (15, 90 * math.pi / 180)
scope = (140, 15)
size = 5
agent = Agent(max_speed, max_stamina, acceleration_limits, scope, size)

# Initial Position
# x = SIZE // 2
# y = SIZE // 2
(x, y) = envn.agent_loc[0]  # a single agent (hider)
print(f'Initial position of the agent: {(x, y)}')
print(f'type(x): {type(x)}')


def init_drawing():
    global show_world
    show_world = cv2.cvtColor((world * 255).astype('uint8'), cv2.COLOR_GRAY2BGR)
    show_world[np.where((show_world == (255, 255, 255)).all(axis=2))] = (255, 0, 0)


# @njit()
def get_line(p1, p2, dist=None):
    x1, y1 = p1
    x2, y2 = p2

    if dist is None:
        dist = int(math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))

    positions = []
    if x1 == x2:  # If grad is inf.
        for yi in np.arange(y1, y2, (y2 - y1) / dist):
            positions.append((x1, yi))
    else:
        grad = (y2 - y1) / (x2 - x1)
        y_intercept = y1 - grad * x1

        for xi in np.arange(x1, x2, (x2 - x1) / dist):
            yi = grad * xi + y_intercept
            positions.append((xi, yi))

    return positions


# JIT on this function makes it a scale faster
# @njit()
def get_vision(env, position, orientation, constraints):
    x_pos, y_pos = position
    angular_scope, dist_scope = constraints
    angular_scope = angular_scope * math.pi / 180

    # Distance at left/right limits
    max_dist = dist_scope / math.cos(angular_scope / 2)

    # Get (relative) left limit
    xl = int(x_pos + max_dist * math.cos(orientation - angular_scope / 2))
    yl = int(y_pos + max_dist * math.sin(orientation - angular_scope / 2))

    # right limit
    xr = int(x_pos + max_dist * math.cos(orientation + angular_scope / 2))
    yr = int(y_pos + max_dist * math.sin(orientation + angular_scope / 2))

    # Arrays to hold vision and distance information
    view = []
    dist = []

    vis_line = get_line((xl, yl), (xr, yr), dist=int(2 * math.sqrt(dist_scope ** 2 + max_dist ** 2)))

    for vis_pt in vis_line:
        vis_pt = (int(vis_pt[0]), int(vis_pt[1]))

        sight_line = get_line((x_pos, y_pos), vis_pt)
        sight_value = [env[int(ye), int(xe)] if env.shape[0] > int(xe) > 0 and env.shape[0] > int(ye) > 0 else 1
                       for xe, ye in sight_line]

        try:
            dist.append(sight_value.index(1))
            view.append(1)
        except Exception:  # ValueError, but numba jit only supports Exception
            dist.append(np.inf)
            view.append(0)

    view = np.asarray(view)
    dist = np.asarray(dist)
    # numba complains, probably due to type differences, just return as tuple for the moment
    # vision = np.stack([view, dist], axis=0)
    return ((xl, yl), (xr, yr)), (view, dist)


# ----------------------------------------------------------------------

# ---- TESTS ----
TEST = 2

if TEST == 1:
    # Random Walk
    while True:
        init_drawing()
        cv2.circle(show_world, (x, y), agent.size, (255, 255, 255), -1)

        # Perform Action
        delta_x, delta_y = agent.action(choice=1, ang_accel=random.randint(-180, 180) * math.pi / 180,
                                        accel=random.randint(0, 5))
        x += delta_x
        y += delta_y

        cv2.imshow("world", show_world)
        k = cv2.waitKey(50)
        if k == 27:
            break

elif TEST == 2:
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('walls.mp4', fourcc, 20.0, (world.shape[1], world.shape[0]))

    # Wall avoiding rule based policy
    while True:
        init_drawing()
        cv2.circle(show_world, (x, y), agent.size, (255, 255, 255), -1)

        # Get agent vision
        (pa, pb), vis = get_vision(world, (x, y), agent.angle, agent.scope)

        # Draw vision
        cv2.imshow("vision", cv2.resize((vis[0] * 255).astype('uint8'), (10, vis[0].shape[0] * 10),
                                        interpolation=cv2.INTER_NEAREST))
        cv2.line(show_world, tuple(int(n) for n in pa), tuple(int(n) for n in pb), (200, 0, 200), 2)

        # Policy
        if 1 in vis[0]:
            ang_acc = random.randint(20, 45)
            acc = -10
        else:
            ang_acc = 0
            acc = 5

        # Action based on Policy
        delta_x, delta_y = agent.action(choice=1, ang_accel=(ang_acc * math.pi / 180), accel=acc)
        x += min(delta_x, world.shape[1] - (5 + agent.size) - x)
        y += min(delta_y, world.shape[0] - (5 + agent.size) - y)

        # Clamp position to the walls if negative
        x = 5 + agent.size if x < 5 + agent.size else x
        y = 5 + agent.size if y < 5 + agent.size else y

        out.write(show_world)
        cv2.imshow("world", show_world)
        k = cv2.waitKey(0)
        if k == 27:
            break

    cv2.destroyAllWindows()
    out.release()
