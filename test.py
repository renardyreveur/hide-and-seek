import math
import random

import cv2
import numpy as np
from numba import njit

from agent import Agent
from environment import Map

# ---- ENVIRONMENT ----
SIZE = 1000
show_world = None
envn = Map(SIZE, SIZE, 20)
world = envn.map

# ---- SINGLE AGENT ----
max_speed = 30
max_stamina = 10

acceleration_limits = (15, 90 * math.pi / 180)
scope = (90, 55)
size = 5
agent1 = Agent(2, max_speed, max_stamina, acceleration_limits, scope, size)
agent2 = Agent(3, max_speed, max_stamina, acceleration_limits, scope, size)

agents = [agent1, agent2]


def init_drawing():
    global show_world, envn, world
    envn.refresh_map()
    world = envn.map
    show_world = cv2.cvtColor((world * 255).astype('uint8'), cv2.COLOR_GRAY2BGR)
    show_world[np.where((show_world == (255, 255, 255)).all(axis=2))] = (255, 0, 0)


@njit()
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
@njit()
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

    # Get visual scope line
    vis_line = get_line((xl, yl), (xr, yr), dist=int(2 * math.sqrt(dist_scope ** 2 + max_dist ** 2)))

    # For every line from the agent to the visual limit
    for vis_pt in vis_line:
        vis_pt = (int(vis_pt[0]), int(vis_pt[1]))

        sight_line = get_line((x_pos, y_pos), vis_pt)
        sight_value = [env[int(ye), int(xe)] if env.shape[1] > int(xe) > 0 and env.shape[0] > int(ye) > 0 else 1
                       for xe, ye in sight_line]

        # Test for object in sight
        whs_query = [np.inf, np.inf, np.inf]
        if 1 in sight_value[2:]:
            whs_query[0] = sight_value[2:].index(1) + 2
        if 2 in sight_value[2:]:
            whs_query[1] = sight_value[2:].index(2) + 2
        if 3 in sight_value[2:]:
            whs_query[2] = sight_value[2:].index(3) + 2

        # If not object in sight, 0(background) in view at inf dist.
        if whs_query[0] == np.inf and whs_query[1] == np.inf and whs_query[2] == np.inf:
            dist.append(np.inf)
            view.append(0)
        else:
            dist.append(min(whs_query))
            view.append(whs_query.index(min(whs_query)) + 1)

    view = np.asarray(view)
    dist = np.asarray(dist)
    # numba complains, probably due to type differences, just return as tuple for the moment
    # vision = np.stack([view, dist], axis=0)
    return ((xl, yl), (xr, yr)), (view, dist)


# ----------------------------------------------------------------------

# ---- TESTS ----
# Initial Position
agent_pos = []
for a in agents:
    agent_pos.append(
        [random.randint(envn.width // 3, envn.width * 2 // 3), random.randint(envn.height // 3, envn.height * 2 // 3)])

history = [[], []]

# Wall avoiding rule based policy
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out = cv2.VideoWriter('walls.mp4', fourcc, 20.0, (world.shape[1], world.shape[0]))
while True:
    init_drawing()
    for i, agt in enumerate(agents):
        x, y = agent_pos[i]
        world[y, x] = agt.agt_class

        # Draw Agent
        if agt.agt_class == 2:
            color = (200, 150, 0)
        else:
            color = (0, 200, 200)
        cv2.circle(show_world, (x, y), agt.size, color, -1)

        # Get agent vision
        (pa, pb), vis = get_vision(world, (x, y), agt.angle, agt.scope)

        # Draw vision
        cv2.imshow("vision", cv2.resize((vis[0] * 255).astype('uint8'), (10, vis[0].shape[0] * 10),
                                        interpolation=cv2.INTER_NEAREST))
        cv2.line(show_world, tuple(int(n) for n in pa), tuple(int(n) for n in pb), (200, 0, 200), 2)

        # Policy
        if 1 in vis[0]:  # If wall in vision, rotate
            action = 1
            action_param = {"ang_accel": (random.randint(20, 45) * math.pi / 180),
                            "accel": -10}

        elif len(set(history[i])) == 1:  # If stationary for 3 time steps rotate
            action = 1
            action_param = {"ang_accel": (random.randint(20, 45) * math.pi / 180),
                            "accel": 0}

        elif agt.agt_class == 3 and 2 in vis[0] and vis[1][list(vis[0]).index(2)] < 60:  # If hider in front, tag
            action = 2
            action_param = {}

        else:  # When there isn't a special event, just move forward
            action = 1
            action_param = {"ang_accel": (0 * math.pi / 180), "accel": 5}

        # Action based on Policy
        delta_x, delta_y = agt.action(choice=action, **action_param)
        delta_x = min(delta_x, world.shape[1] - (5 + agt.size) - x)
        delta_y = min(delta_y, world.shape[0] - (5 + agt.size) - y)

        if envn is not None:
            # Agent cannot cross walls
            for w in envn.wall_loc:
                w = w[::-1]
                # cv2.circle(show_world, tuple(int(ee) for ee in w), 3, (0, 0, 255), -1)
                if delta_x >= 0 and delta_y >= 0:
                    if x <= w[0] <= x + delta_x and y <= w[1] <= y + delta_y:
                        delta_x, delta_y = 0, 0
                elif delta_x >= 0 >= delta_y:
                    if x <= w[0] <= x + delta_x and y >= w[1] >= y + delta_y:
                        delta_x, delta_y = 0, 0
                elif delta_x <= 0 <= delta_y:
                    if x >= w[0] >= x + delta_x and y <= w[1] <= y + delta_y:
                        delta_x, delta_y = 0, 0
                elif delta_x <= 0 and delta_y <= 0:
                    if x >= w[0] >= x + delta_x and y >= w[1] >= y + delta_y:
                        delta_x, delta_y = 0, 0

        agent_pos[i][0] += delta_x
        agent_pos[i][1] += delta_y

        # Clamp position to the walls if negative
        agent_pos[i][0] = 5 + agt.size if agent_pos[i][0] < 5 + agt.size else agent_pos[i][0]
        agent_pos[i][1] = 5 + agt.size if agent_pos[i][1] < 5 + agt.size else agent_pos[i][1]

        history[i].append((x, y))
        if len(history[i]) > 3:
            history[i] = history[i][1:]

    # out.write(show_world)
    cv2.imshow("world", show_world)
    k = cv2.waitKey(1)
    if k == 27:
        break

cv2.destroyAllWindows()
# out.release()
