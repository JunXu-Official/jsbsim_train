import math
from ..core.catalog import Catalog as c
from .termination_condition_base import BaseTerminationCondition


class UnreachAltitude(BaseTerminationCondition):
    """
    UnreachAltitude [0, 1]
    End up the simulation if the aircraft didn't reach the target altitude in limited time.
    """

    def __init__(self, config):
        super().__init__(config)
        uid = list(config.aircraft_configs.keys())[0]
        aircraft_config = config.aircraft_configs[uid]
        self.max_altitude_increment = aircraft_config['max_altitude_increment']   #高度最多变化2133.6m = 7000ft
        self.check_interval = aircraft_config['check_interval']   #每30s判断一次是否达到期望值
        self.increment_size = [0.2, 0.4, 0.6, 0.8, 1.0] + [1.0] * 10

    def get_termination(self, task, env, agent_id, info={}):
        """
               判断飞机在限定时间内是否达到目标高度，如没有，则终止仿真；否则，生成新的目标高度等参数继续训练
        Args:
            task: task instance
            env: environment instance

        Returns:Q
            (tuple): (done, success, info)
        """
        done = False
        success = False
        cur_step = info['current_step']
        check_time = env.agents[agent_id].get_property_value(c.altitude_check_time)   #检查航向的时间点
        # check altitude when simulation_time exceed check_time
        if env.agents[agent_id].get_property_value(c.simulation_sim_time_sec) >= check_time:
            if math.fabs(env.agents[agent_id].get_property_value(c.delta_altitude)) > 300:   #当前航向偏差超过10°，视为没达到目标   c---->catalog.py
                done = True
            # if current target altitude is reached, random generate a new target altitude
            else:

                delta = self.increment_size[env.altitude_turn_counts]
                delta_altitude = env.np_random.uniform(-delta, delta) * self.max_altitude_increment

                new_altitude = env.agents[agent_id].get_property_value(c.target_altitude_ft) + delta_altitude
                new_altitude = max(new_altitude, 15000) # assert the value in safe region

                env.agents[agent_id].set_property_value(c.target_altitude_ft, new_altitude)
                env.agents[agent_id].set_property_value(c.altitude_check_time, check_time + self.check_interval)

                env.altitude_turn_counts += 1
                self.log(f'current_step:{cur_step} 'f'target_altitude_ft:{new_altitude}')
        if done:
            self.log(f'agent[{agent_id}] unreached altitude. Total Steps={env.current_step}')
            info['altitude_turn_counts'] = env.altitude_turn_counts
        success = False
        return done, success, info
