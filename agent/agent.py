

class Agent:
    def __init__(self, max_speed, max_stamina, acceleration, visual_scope, size):
        # State history
        self.history = []

        # Movement constraints
        self.max_speed = max_speed
        self.max_stamina = max_stamina
        self.acceleration = acceleration

        # Movement State
        self.angle = 0
        self.speed = 0
        self.stamina = max_stamina

        # Observation
        self.scope = visual_scope

        # Intrinsic State
        self.size = size

    def observe_env(self):
        """
        Get information about the environment within the visual scope
        :return:
        """
        # TODO: How am I going to query the environment for a 1D view of my position at my current angle?
        # The agent shouldn't know about it's absolute position in the environment, but only the relative position.
        pass

    def communicate(self):
        """
        Send a 'near-by' signal to the same team agents
        """
        self.observe_env()

    def move(self, angle, speed):
        """
        Given an angle and speed, attempt to move accordingly, constrained by acceleration and stamina
        :param angle: Target angle
        :param speed: Target sped
        """
        self.observe_env()

    def tag(self):
        """
        For hiders: Tag the checkpoint
        For seekers: Tag the hiders
        """
        self.observe_env()

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
        act = actions.get(choice)
        act(*args, **kwargs)
