import cv2

from environment.env import World

# ---- TEST Parameters ----
MAX_TIMESTEP = 5

# ---- World Gen. Parameters ----
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

# Generate our test WORLD!!!
show_world = None
new_world = World(agent_cfg, agent_kwargs,
                  map_size, max_num_walls, borders, sound_lim)

world = new_world.map
new_world.init_drawing()

# Iterate over time!
for t in range(MAX_TIMESTEP):
    # world = new_world.map
    print(f't: {t}')
    for i, agt in enumerate(new_world.agents):
        new_world.update()

        x, y = new_world.agent_loc[i]
        world[y, x] = agt.agt_class

        # Draw Agent
        if agt.agt_class == 2:
            color = (200, 150, 0)
        else:
            color = (0, 200, 200)

        cv2.circle(show_world, (x, y), agt.size, color, -1)

        # Get agent vision
        print(f'vision: {agt.vision}')
        (pa, pb), vis = agt.vision

        # Draw vision
        cv2.imshow('vision', cv2.resize((vis[0]*255).astype('uint8'), (10, vis[0].shape[0] * 10),
                                        interpolation=cv2.INTER_NEAREST))
        cv2.line(show_world, tuple(int(n) for n in pa), tuple(int(n) for n in pb), (200, 0, 200), 2)


    cv2.imshow("WORLD!", show_world)
    cv2.waitKey(0)



cv2.destroyAllWindows()
