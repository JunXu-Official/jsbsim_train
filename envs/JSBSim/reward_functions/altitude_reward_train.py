import math
from .reward_function_base import BaseRewardFunction
from ..core.catalog import Catalog as c


class AltitudeRewardTrain(BaseRewardFunction):
    """
    Measure the difference between the current altitude and the target altitude
    """
    def __init__(self, config):
        super().__init__(config)
        self.reward_item_names = [self.__class__.__name__ + item for item in ['', '_alt']]

    def get_reward(self, task, env, agent_id):
        """
        Reward is built as a geometric mean of scaled gaussian rewards for each relevant variable

        Args:
            task: task instance
            env: environment instance

        Returns:
            (float): reward
        """

        #高度奖励
        alt_error_scale = 50
        alt_r = math.exp(-((env.agents[agent_id].get_property_value(c.delta_altitude) / alt_error_scale) ** 2))

        reward = alt_r ** (1 / 4)
        return self._process(reward, agent_id, alt_r)
