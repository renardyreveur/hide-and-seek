import math

import numpy as np
from agent import Agent
import cv2
import random

SIZE = 1000
world = np.zeros((SIZE, SIZE))
show_world = None
# Basic walls
world[0:5, :] = 1
world[SIZE - 6: SIZE - 1, :] = 1
world[:, 0:5] = 1
world[:, SIZE - 6: SIZE - 1] = 1

# Single agent
max_speed = 30
max_stamina = 10
acceleration_limits = (15, 90 * math.pi / 180)
scope = (100, 30)
size = 5
agent = Agent(max_speed, max_stamina, acceleration_limits, scope, size)

# Initial Position
x = SIZE // 2
y = SIZE // 2


def init_drawing():
    global show_world
    show_world = cv2.cvtColor((world * 255).astype('uint8'), cv2.COLOR_GRAY2BGR)
    show_world[0:5, :] = (255, 0, 0)
    show_world[SIZE - 6: SIZE - 1, :] = (255, 0, 0)
    show_world[:, 0:5] = (255, 0, 0)
    show_world[:, SIZE - 6: SIZE - 1] = (255, 0, 0)


def get_vision(env, position, orientation, constraints):
    x_pos, y_pos = position
    angular_scope, dist_scope = constraints
    angular_scope = angular_scope * math.pi / 180

    # Distance at left/right limits
    max_dist = dist_scope / math.cos(angular_scope / 2)

    # Get (relative) left limit
    x1 = x_pos + max_dist * math.cos(orientation + angular_scope / 2)
    y1 = y_pos + max_dist * math.sin(orientation + angular_scope / 2)

    # right limit
    x2 = x_pos + max_dist * math.cos(orientation - angular_scope / 2)
    y2 = y_pos + max_dist * math.sin(orientation - angular_scope / 2)

    # Arrays to hold vision and distance information
    view = []
    dist = []
    if x1 == x2:  # If grad is inf.
        for yi in range(int(y1), int(y2), -1 if y1 > y2 else 1):
            try:
                view.append(env[int(yi), int(x1)])
                dist.append(math.sqrt((x_pos - x1) ** 2 + (y_pos - yi) ** 2))
            except IndexError:
                view.append(1)
                dist.append(math.sqrt((x_pos - x1) ** 2 + (y_pos - yi) ** 2))
    else:
        grad = abs(y2 - y1) / ((x2 - x1) if x2 > x1 else (x1 - x2))
        y_intercept = y1 - int(grad * x1)

        for xi in range(int(x1), int(x2), -1 if x1 > x2 else 1):
            yi = int(grad * xi + y_intercept)
            try:
                if yi < 0 or xi < 0:
                    raise IndexError  # as python supports negative indexing
                view.append(env[yi, xi])
                dist.append(math.sqrt((x_pos - xi) ** 2 + (y_pos - yi) ** 2))
            except IndexError:
                view.append(1)
                dist.append(math.sqrt((x_pos - xi) ** 2 + (y_pos - yi) ** 2))

    view = np.asarray(view)
    dist = np.asarray(dist)
    vision = np.stack([view, dist], axis=0)
    return ((x1, y1), (x2, y2)), vision


# ----------------------------------------------------------------------
# Choice of test
TEST = 2

if TEST == 1:
    # Random Walk
    while True:
        init_drawing()
        cv2.circle(show_world, (x, y), agent.size, (255, 255, 255), -1)
        delta_x, delta_y = agent.action(choice=1, ang_accel=random.randint(-180, 180) * math.pi / 180,
                                        accel=random.randint(0, 5))
        x += delta_x
        y += delta_y
        cv2.imshow("world", show_world)
        k = cv2.waitKey(200)
        if k == 27:
            break

elif TEST == 2:
    # Wall avoiding rule based policy
    while True:
        init_drawing()
        cv2.circle(show_world, (x, y), agent.size, (255, 255, 255), -1)

        # Get agent vision
        (pa, pb), vis = get_vision(world, (x, y), agent.angle, agent.scope)
        cv2.line(show_world, tuple(int(n) for n in pa), tuple(int(n) for n in pb), (200, 0, 200), 2)

        # Policy
        if 1 in vis[0]:
            ang_acc = 30
            acc = -10
        else:
            ang_acc = 0
            acc = 5

        # Action
        delta_x, delta_y = agent.action(choice=1, ang_accel=(ang_acc * math.pi / 180), accel=acc)
        x += min(delta_x, SIZE - x)
        y += min(delta_y, SIZE - y)

        # Clamp position to 0 if negative
        x = 0 if x < 0 else x
        y = 0 if y < 0 else y

        cv2.imshow("world", show_world)
        k = cv2.waitKey(100)
        if k == 27:
            break
    cv2.destroyAllWindows()
