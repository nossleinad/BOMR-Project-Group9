"""Connection to the Thymio: the only place that talks to the robot."""
import numpy as np
from tdmclient import ClientAsync


class Thymio:
    VARIABLES = {"prox.horizontal", "motor.left.speed", "motor.right.speed", "button.center"}

    def __init__(self, client, node):
        self.client, self.node = client, node

    @classmethod
    async def connect(cls):
        """Connect to the first Thymio found, lock it and wait for its sensors."""
        client = ClientAsync()
        node = await client.wait_for_node()
        await node.lock()
        await node.wait_for_variables(cls.VARIABLES)
        return cls(client, node)

    # sensors
    def read_prox(self):
        return list(self.node["prox.horizontal"])

    def read_wheel_speeds(self):
        return np.array([self.node["motor.left.speed"], self.node["motor.right.speed"]], dtype=float)

    def stop_requested(self):
        """True when the middle button is pressed (emergency stop)."""
        return self.node["button.center"] == 1

    # actuators
    def set_wheel_speeds(self, left, right):
        self.node.send_set_variables({"motor.left.target": [int(left)],
                                      "motor.right.target": [int(right)]})

    def stop(self):
        self.set_wheel_speeds(0, 0)

    # timing and closing
    async def sleep(self, dt):
        """Wait dt seconds; also receives the new sensor values."""
        await self.client.sleep(dt)

    async def close(self):
        """Stop the motors and unlock the robot."""
        await self.node.set_variables({"motor.left.target": [0], "motor.right.target": [0]})
        await self.node.unlock()