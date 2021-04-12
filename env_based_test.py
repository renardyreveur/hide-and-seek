import cv2

from environment import World

# ---- TEST Parameters ----
MAX_TIMESTEP = 10000

# ---- World Gen. Parameters ----
agent_cfg = (2, 2)                # num. hiders, num. seekers
agent_kwargs = {
    "max_speed": 10,              # max speed of an agent
    "max_stamina": 10,            # max stamina of an agent
    "accel_limit": (45, 5),       # acceleration limit (angular, linear)
    "visual_scope": (90, 55),     # visual scope (angular, max_distance)
    "size": 5                     # agent size
}

map_size = (1000, 1000)           # maximum map size
max_num_walls = 20                # maximum number of walls
borders = True                    # border settings of the map


# Generate our test WORLD!!!
new_world = World(agent_cfg, agent_kwargs,
                  map_size, max_num_walls, borders)

# Iterate over time!
for t in range(MAX_TIMESTEP):
    new_world.update()

    show_world = new_world.map
    cv2.imshow("WORLD!", show_world)
    cv2.waitKey(0)

cv2.destroyAllWindows()
