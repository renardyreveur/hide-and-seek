import random
import sys

import asyncio
import websockets
import time

sys.path.append("./")
import game_state_pb2 as GameState
from environment import World


# ---- World Gen. Parameters ----
map_type = "amongUs"

agent_cfg = (2, 2)                # num. hiders, num. seekers
agent_kwargs = {
    "max_speed": 10,              # max speed of an agent
    "max_stamina": 10,            # max stamina of an agent
    "accel_limit": (45, 5),       # acceleration limit (angular, linear)
    "visual_scope": (90, 55),     # visual scope (angular, max_distance)
    "size": 5,                    # agent size
    "count": 0
}

map_size = (1000, 1000)           # maximum map size
max_num_walls = 20                # maximum number of walls
borders = True                    # border settings of the map
sound_lim = 90                    # all sounds above 90 dB are damaging the inner ear


async def game_state(websocket, path):
    # Generate our test WORLD!!!
    world = World(map_type, agent_cfg, agent_kwargs, map_size, max_num_walls, borders, sound_lim)
    map_h, map_w = world.map.shape[:2]
    wall_loc = [GameState.Point(x=x, y=y) for (x, y) in world.wall_loc]

    start = time.time()
    while True:
        current_time = time.time()
        delta_time = current_time - start
        if delta_time > 0.03:
            world.update()
            agents = world.agents
            a_class = [x.agt_class for x in agents]
            a_loc = world.agent_loc
            agent_bytes = [GameState.Agent(uid=a.uid,
                                           agent_class=a_class[a.uid],
                                           location=GameState.Point(**dict(zip(("x", "y"), a_loc[a.uid]))))
                           for a in agents]

            gs = GameState.GameState(walls=wall_loc,
                                     agents=agent_bytes,
                                     mapsize=GameState.Size(width=map_w, height=map_h))
            print(delta_time)
            await websocket.send(gs.SerializeToString())
            print("HERE")
            start = time.time()

start_server = websockets.serve(game_state, "localhost", 7393)
asyncio.get_event_loop().run_until_complete(start_server)
print("Game Server Started!!")
asyncio.get_event_loop().run_forever()


# # Get update results per agent
# for i, agt in enumerate(world.agents):
#     # Get agent location to draw
#     x, y = world.agent_loc[i]
#
#     # Draw Agent
#     color = (200, 150, 0) if agt.agt_class == 2 else (0, 200, 200)
#     cv2.circle(show_world, (x, y), agt.size, color, 2)
#
#     # Draw Agent Vision
#     (pa, pb), _, _ = agt.vision
#     cv2.line(show_world, tuple(int(n) for n in pa), tuple(int(n) for n in pb), (200, 0, 200), 2)
