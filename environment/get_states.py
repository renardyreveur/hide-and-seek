import math

import numpy as np


def get_line(p1, p2, dist=None):
    x1, y1 = p1
    x2, y2 = p2

    if dist is None:
        dist = int(math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))

    positions = []
    if x1 == x2:
        for yi in np.arange(y1, y2, (y2 - y1) / dist):
            positions.append((x1, yi))
    else:
        grad = (y2 - y1) / (x2 - x1)
        y_intercept = y1 - grad * x1

        for xi in np.arange(x1, x2, (x2 - x1) / dist):
            yi = grad * xi + y_intercept
            positions.append((xi, yi))
    return positions


def get_vision(env, position, orientation, scope):
    x_pos, y_pos = position
    angular_scope, dist_scope = scope
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
        # Sight limited by the dist scope
        sight_line = get_line((x_pos, y_pos), vis_pt)[:dist_scope]
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
    # TODO: match return to agent vision attribute
    return ((xl, yl), (xr, yr)), (view, dist)


def get_sound():
    pass


def get_communication():
    pass
