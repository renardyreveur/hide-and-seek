import math

import numpy as np


# Get coordinates that form a line between two lines
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
    sight_c = []
    # Get visual scope line
    vis_line = get_line((xl, yl), (xr, yr), dist=int(2 * math.sqrt(dist_scope ** 2 + max_dist ** 2)))

    # For every line from the agent to the visual limit
    for vis_pt in vis_line:
        vis_pt = (int(vis_pt[0]), int(vis_pt[1]))
        # Sight limited by the dist scope
        sight_line = get_line((x_pos, y_pos), vis_pt)[:dist_scope]
        sight_value = [env[int(ye), int(xe)] if env.shape[1] > int(xe) > 0 and env.shape[0] > int(ye) > 0 else 1
                       for xe, ye in sight_line]
        sight_coord = [(int(xe), int(ye)) if env.shape[1] > int(xe) > 0 and env.shape[0] > int(ye) > 0 else 1
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
            sight_c.append(sight_coord)
        else:
            dist.append(min(whs_query))
            view.append(whs_query.index(min(whs_query)) + 1)
            sight_c.append(sight_coord)
    view = np.asarray(view)
    dist = np.asarray(dist)
    # print(f'xl: {xl}, yl: {yl}')
    # numba complains, probably due to type differences, just return as tuple for the moment
    # vision = np.stack([view, dist], axis=0)
    # TODO: match return to agent vision attribute
    return ((xl, yl), (xr, yr)), (view, dist), sight_c


# TODO: 그 딱 경계에 있을 때 처리해주
def get_sound(locs, agents, agent_id, sound_limit):
    # position of the agent_i
    pos_x, pos_y = locs[agent_id]
    vision_ang = agents[agent_id].angle

    # agent_i가 바라보고 있는 방향의 벡터
    v_x, v_y = (math.cos(vision_ang), math.sin(vision_ang))

    # positions of the other agents
    # strength of the sound = 1/(distance**2)
    dist = []
    orient = []
    for i in range(len(locs)):
        if i != agent_id:
            p_x, p_y = locs[i]
            dist.append((p_x - pos_x) ** 2 + (p_y - pos_y) ** 2)

            # 나로부터 i번째 agent의 포지션으로의 벡터
            vi_x, vi_y = (p_x - pos_x, p_y - pos_y)

            # Direct way to computing clockwise angle between 2 vectors
            # Dot product is proportional to the cosine of the angle, the determinant is proportional to its sine.
            dot = v_x * vi_x + v_y * vi_y
            det = v_x * vi_y - v_y * vi_x

            # The atan2() function returns a value in the range -pi to pi radians.
            ang_i = math.atan2(det, dot)

            if -math.pi <= ang_i < -0.75 * math.pi:
                orient.append(1)
            elif -0.75 * math.pi <= ang_i < -0.5 * math.pi:
                orient.append(2)
            elif -0.5 * math.pi <= ang_i < -0.25 * math.pi:
                orient.append(3)
            elif -0.25 * math.pi <= ang_i < 0:
                orient.append(4)
            elif 0 <= ang_i < 0.25 * math.pi:
                orient.append(5)
            elif 0.25 * math.pi <= ang_i < 0.5 * math.pi:
                orient.append(6)
            elif 0.5 * math.pi <= ang_i < 0.75 * math.pi:
                orient.append(7)
            else:
                orient.append(8)

    # all floats to 4 decimal places
    decibel = [round(1 / i, 4) for i in dist]

    assert len(decibel) == len(orient)
    filtered = [(s, o) for s, o in list(zip(decibel, orient)) if s > sound_limit]

    return filtered


# get_sound에서의 dist는 sound limit 안에 해당하는 에이전트에 대해서만 strength 계산에 있어 사용되는 반면,
# get_communication은 제한이 없고, 절대적인 distance를 반환한다.
# 또, get_sound는 자기 중심으로 다른 에이전트의 방향을 대략적으로 알게끔 하지만,
# get_comm에서의 orient는 다른 에이전트로 부터 get_comm을 호출한 에이전트로의 방향을 안다는 점에서 orient의 의미에서 차이가 있다.
# 남아있는 모든 에이전트에게 정보를 보내는 것이기 때문에 에이전트 개수만큼을 반환해서 각 에이전트가 i번째 인자를 가져가도록 하게끔 하자
# 죽은 에이전트는 죽었으니 정지한 상태일까? 어쨋든 죽었든 살았든 반환개수는 에이전트 개수만큼 하는 게 좋을 것 같은데
# 만약 카운트(get comm 할 수 있는 횟수)를 다 썼으면 아무것도 반환하지 않는다.

def send_communication(locs, agents, agent_id):
    agent = next((x for x in agents if x.uid == agent_id), None)

    # position of the agent_i
    pos_x, pos_y = locs[agent_id]

    for uid, location in locs.items():
        # Agent in question
        aiq = next((x for x in agents if x.uid == uid), None)

        # If yourself, or other team, skip
        if uid == agent_id or aiq.agt_class != agent.agt_class:
            continue

        # Get distance to the other agent
        p_x, p_y = location
        distance = (p_x - pos_x) ** 2 + (p_y - pos_y) ** 2

        # Get direction of the agent (from it's view to the other agent's location)
        vision_ang_i = aiq.angle
        v_x, v_y = (math.cos(vision_ang_i), math.sin(vision_ang_i))
        vi_x, vi_y = (pos_x - p_x, pos_y - p_y)

        # Direct way to compute clockwise angle between 2 vectors
        # Dot product is proportional to the cosine of the angle, the determinant is proportional to its sine.
        dot = v_x * vi_x + v_y * vi_y
        det = v_x * vi_y - v_y * vi_x

        # The atan2() function returns a value in the range -pi to pi radians.
        ang_i = math.atan2(det, dot)
        bearing = ang_i

        # Update all the other agent's states with the communication record
        aiq.comm.append([distance, bearing])

