import cv2

from environment import World

# ---- TEST Parameters ----
MAX_TIMESTEP = 300
SAVE_VID = False

# ---- World Gen. Parameters ----
map_type = "empty_map"


agent_cfg = (0, 2)                # num. hiders, num. seekers
agent_kwargs = {
    "max_speed": 10,              # max speed of an agent
    "max_stamina": 10,            # max stamina of an agent
    "accel_limit": (45, 5),       # acceleration limit (angular, linear)
    "visual_scope": (90, 55),     # visual scope (angular, max_distance)
    "size": 5,                    # agent size
    "comm_limit": 10
}


map_size = (1200, 1200)     # maximum map size
max_num_walls = 20          # maximum number of walls - large number required for the voronoi map(100), random_walk
borders = True              # border settings of the map
sound_lim = 90              # all sounds above 90 dB are damaging the inner ear


# Generate our test WORLD!!!
world = World(map_type, agent_cfg, agent_kwargs, map_size, max_num_walls, borders, sound_lim)

# To save a video
video_size = world.map.shape[:2][::-1]
v_writer = cv2.VideoWriter("test.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 20, video_size) if SAVE_VID else None


# Iterate over time!
for t in range(MAX_TIMESTEP):
    print(f'\nTIME: {t}')

    # Initialise map canvas to draw
    show_world = world.init_drawing()

    # World update!
    world.update()

    # Get update results per agent
    for agt in world.agents:
        # Get agent location to draw
        x, y = world.agent_loc[agt.uid]

        # Draw Agent
        color = (200, 150, 0) if agt.agt_class == 2 else (0, 200, 200)
        cv2.circle(show_world, (x, y), agt.size, color, 2)

        # Draw Agent Vision
        (pa, pb), _, _ = agt.vision
        cv2.line(show_world, tuple(int(n) for n in pa), tuple(int(n) for n in pb), (200, 0, 200), 2)

    if SAVE_VID:
        v_writer.write(show_world)

    cv2.imshow("WORLD!", show_world)
    cv2.imshow("map", world.map)
    k = cv2.waitKey(0)
    if k == 27:
        break

cv2.destroyAllWindows()

if SAVE_VID:
    v_writer.release()
