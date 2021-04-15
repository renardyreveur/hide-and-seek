# Ideas

## Agent Communication

- A specified(or unspecified) number of attempts that the agent can use to signal itself to others in the same team

- Maybe the other agents will learn to understand 'no communication for a while' == "Death of an agent"

- Maybe the communication can include the agent index of the same team (or not).
  
  Including the agent index (which is still realistic as you're in the same team) will help with the tracking of the life/deaths of others in the team.
  
  Excluding it, will make it more interesting, and the agent will have to filter through communications carefully to make any decisions.

- The point of the communication action is that you yourself don't get to know your absolute position, but others in the team gets to know your relative position to themselves. Therefore, this action might seem like it has no direct impact on your performance, but who knows? maybe the agents will understand to use this is some manner that will be useful for the whole team.

- Including a cool time for communication might be useful too!

- The communication direction should be relative to each of the agent's position

## Sound

- Sound is a property that is constantly being observed.

- Unlike communication, all agents have to possibility to 'hear' all the other agents' sound, as long as they are close enough (as long as the sound level is greater than some lower limit).

- At one point in time, agents will get the information of N sound sources, N dependent on the whether or not the agent can hear the others. This might help with understanding how many agents there are in the field, etc.

- The sound direction should be relative to the agent's orientation. The specific direction is not given, but one of 8 directions.