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
world[SIZE-6: SIZE-1, :] = 1
world[:, 0:5] = 1
world[:, SIZE-6: SIZE-1] = 1

# Single agent
max_speed = 5
max_stamina = 10
acceleration = (1.1, 0.4)
scope = (100, 30)
size = 5
agent = Agent(max_speed, max_stamina, acceleration, scope, size)

# Initial Position
x = SIZE//2
y = SIZE//2


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

    max_dist = dist_scope/math.cos(angular_scope/2)
    neg_vert_ang = abs(orientation - (angular_scope/2))
    pos_vert_ang = abs(orientation + (angular_scope/2))

    x1 = x_pos - max_dist * math.sin(neg_vert_ang)
    y1 = y_pos + max_dist * math.cos(neg_vert_ang)

    x2 = x_pos + max_dist * math.sin(pos_vert_ang)
    y2 = y_pos + max_dist * math.cos(pos_vert_ang)

    return (x1, y1), (x2, y2)


# Choice of test
TEST = 2

if TEST == 1:
    # Random Walk
    while True:
        init_drawing()
        cv2.circle(show_world, (x, y), agent.size, (255, 255, 255), -1)
        delta_x, delta_y = agent.action(choice=1, angle=random.randint(-180, 180), speed=random.randint(0, max_speed))
        x += delta_x
        y += delta_y
        cv2.imshow("world", show_world)
        k = cv2.waitKey(3)
        if k == 27:
            break

elif TEST == 2:
    # Wall avoiding rule based policy
    while True:
        init_drawing()
        cv2.circle(show_world, (x, y), agent.size, (255, 255, 255), -1)
        delta_x, delta_y = agent.action(choice=1, angle=30, speed=3)
        x += delta_x
        y += delta_y

        # Get agent vision
        print(f"Agent States - angle: {agent.angle}")
        pa, pb = get_vision(world, (x, y), agent.angle, agent.scope)
        cv2.line(show_world, tuple(int(n) for n in pa), tuple(int(n) for n in pb), (200, 0, 200), 2)

        # agent.observe_env(vision)
        cv2.imshow("world", show_world)
        k = cv2.waitKey(100)
        if k == 27:
            break
    cv2.destroyAllWindows()
