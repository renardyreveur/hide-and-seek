import math
from typing import Tuple


class Agent:
    def __init__(self, agt_class: int, max_speed: float, max_stamina: float, accel_limit: Tuple[float, float],
                 visual_scope: Tuple[int, float], size: int):
        # Agent Class 2 - hider, 3 - seeker
        self.agt_class = agt_class
        assert self.agt_class in [2, 3]

        # State history
        self.history = []

        # Movement constraints
        self.max_speed = max_speed
        self.max_stamina = max_stamina
        self.accel_limit = accel_limit

        # Movement State
        self.angle = 0
        self.speed = 0
        self.stamina = max_stamina

        # Observation
        self.scope = visual_scope

        # Intrinsic State
        self.size = size

        # Vision data (updates by observe_env)
        self.vision = None

    def observe_env(self):
        """
        Get information about the environment within the visual scope
        :return: Array with two rows, first indicating vision, the second distance
        """
        # TODO: How am I going to query the environment for a 1D view of my position at my current angle?
        # The agent shouldn't know about it's absolute position in the environment, but only the relative position.
        pass

    def communicate(self):
        """
        Send a 'near-by' signal to the same team agents
        """
        self.observe_env()

    def move(self, ang_accel, accel):
        """
        Given an angle and speed, attempt to move accordingly, constrained by acceleration and stamina
        :param ang_accel: angular acceleration
        :param accel: acceleration
        :return: change in relative x, relative y
        """

        # Get acceleration limits
        acc_limit, ang_acc_limit = self.accel_limit

        # Get change in angle and speed due to the new acceleration given
        delta_angle = min(ang_acc_limit, ang_accel) if ang_accel > 0 else max(-ang_acc_limit, ang_accel)
        delta_speed = min(acc_limit, accel) if accel > 0 else max(accel, acc_limit)
        # print(f"accel speed: {accel}, accel_angle: {ang_accel}")
        # print(f"accel limit: {acc_limit}, angle_limit: {ang_acc_limit}")
        # print(f"delta speed: {delta_speed}, delta_angle: {delta_angle}")

        # Calculate new angle and speed of agent
        self.angle += delta_angle
        self.angle %= (2*math.pi)
        self.speed += min(delta_speed, self.max_speed - self.speed)
        # print(f"REAL speed: {self.speed}, REAL angle: {self.angle}")

        # Move the agent accordingly
        delta_x = int(self.speed * math.cos(self.angle))
        delta_y = int(self.speed * math.sin(self.angle))

        self.observe_env()

        return delta_x, delta_y

    def tag(self):
        """
        For hiders: Tag the checkpoint
        For seekers: Tag the hiders
        """
        print("TAGGG!!!!!")
        self.observe_env()
        return 0, 0

    def action(self, choice, *args, **kwargs):
        """
        At every time step an action is performed.
        The action is one of three {"move", "tag", "communicate"}
        Every action ends with an agent state update (observation)
        :param choice: Integer, choice between the three possible actions
        :param args: Arguments for the chosen action
        :param kwargs: Keyword Arguments for the chosen action
        :return:
        """
        actions = {
            1: self.move,
            2: self.tag,
            3: self.communicate
        }
        if self.agt_class == 3:
            print(actions.get(choice).__name__, args, kwargs)

        act = actions.get(choice)
        return act(*args, **kwargs)
