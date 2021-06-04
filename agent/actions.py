import math


def move(agent, ang_accel=3, accel=1):
    """
    Given an angle and speed, attempt to move accordingly, constrained by acceleration and stamina
    :param agent: The agent that is taking the action
    :param ang_accel: angular acceleration
    :param accel: acceleration
    :return: change in relative x, relative y, tag, comm
    """

    # Consider agent stamina
    if abs(agent.speed) < 5:
        agent.stamina += 1 if agent.stamina < agent.max_stamina else 0
    else:
        agent.stamina -= 1 if agent.stamina > 0 else 0

    st_decay = agent.stamina / agent.max_stamina

    # Get acceleration limits
    acc_limit, ang_acc_limit = agent.accel_limit

    # Stamina affects limit
    max_speed = agent.max_speed
    max_speed *= st_decay

    # Get change in angle and speed due to the new acceleration given
    delta_angle = min(ang_acc_limit, ang_accel) if ang_accel > 0 else max(-ang_acc_limit, ang_accel)
    delta_speed = min(acc_limit, accel) if accel >= 0 else max(accel, -acc_limit)
    # print(f"accel speed: {accel}, accel_angle: {ang_accel}")
    # print(f"accel limit: {acc_limit}, angle_limit: {ang_acc_limit}")
    # print(f"delta speed: {delta_speed}, delta_angle: {delta_angle}")

    # Calculate new angle and speed of agent
    agent.angle += delta_angle
    agent.angle %= (2 * math.pi)
    agent.speed += min(delta_speed, max_speed - agent.speed)
    # print(f"REAL speed: {agent.speed}, REAL angle: {agent.angle}")

    # Move the agent accordingly
    delta_x = int(agent.speed * math.cos(agent.angle))
    delta_y = int(agent.speed * math.sin(agent.angle))

    # print(f"delta_x: {delta_x}, delta_y: {delta_y}")
    return delta_x, delta_y, False, False


def tag(agent):
    """
    For hiders: Tag the checkpoint
    For seekers: Tag the hiders
    """
    agent.stamina += 1 if agent.stamina < agent.max_stamina else 0
    return 0, 0, True, False


def communicate(agent):
    """
    Send a 'near-by' signal to the same team agents
    """
    agent.stamina += 1 if agent.stamina < agent.max_stamina else 0
    return 0, 0, False, True
