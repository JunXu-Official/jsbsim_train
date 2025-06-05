import math
from ..core.catalog import Catalog as c
from .termination_condition_base import BaseTerminationCondition


class UnreachHeading(BaseTerminationCondition):
    """
    UnreachHeading [0, 1]
    End up the simulation if the aircraft didn't reach the target heading or attitude in limited time.
    """

    def __init__(self, config):
        super().__init__(config)
        uid = list(config.aircraft_configs.keys())[0]
        aircraft_config = config.aircraft_configs[uid]
        self.max_heading_increment = aircraft_config['max_heading_increment']     #航向最大增加180
        self.max_altitude_increment = aircraft_config['max_altitude_increment']   #高度最多变化2133.6m = 7000ft
        self.max_velocities_u_increment = aircraft_config['max_velocities_u_increment']     # 速度最多变化100
        self.check_interval = aircraft_config['check_interval']   #每30s判断一次是否达到期望值
        self.increment_size = [0.2, 0.4, 0.6, 0.8, 1.0] + [1.0] * 10

    def get_termination(self, task, env, agent_id, info={}):
        """
               判断飞机在限定时间内是否达到目标航向，如没有，则终止仿真；否则，生成新的目标航向等参数继续训练
        Args:
            task: task instance
            env: environment instance

        Returns:Q
            (tuple): (done, success, info)
        """
        done = False
        success = False
        cur_step = info['current_step']
        check_time = env.agents[agent_id].get_property_value(c.heading_check_time)   #检查航向的时间点
        # check heading when simulation_time exceed check_time
        if env.agents[agent_id].get_property_value(c.simulation_sim_time_sec) >= check_time:
            if math.fabs(env.agents[agent_id].get_property_value(c.delta_heading)) > 10:   #当前航向偏差超过10°，视为没达到目标   c---->catalog.py
                done = True
            # if current target heading is reached, random generate a new target heading
            else:

                delta = self.increment_size[env.heading_turn_counts]
                delta_heading = env.np_random.uniform(-delta, delta) * self.max_heading_increment
                delta_altitude = env.np_random.uniform(-delta, delta) * self.max_altitude_increment
                delta_velocities_u = env.np_random.uniform(-delta, delta) * self.max_velocities_u_increment
                new_heading = env.agents[agent_id].get_property_value(c.target_heading_deg) + delta_heading
                new_heading = (new_heading + 360) % 360
                new_altitude = env.agents[agent_id].get_property_value(c.target_altitude_ft) + delta_altitude
                new_altitude = max(new_altitude, 15000) # assert the value in safe region
                new_velocities_u = env.agents[agent_id].get_property_value(c.target_velocities_u_mps) + delta_velocities_u
                new_velocities_u = max(new_velocities_u, 120.) # assert the value in safe region

                env.agents[agent_id].set_property_value(c.target_heading_deg, new_heading)
                env.agents[agent_id].set_property_value(c.target_altitude_ft, new_altitude)
                env.agents[agent_id].set_property_value(c.target_velocities_u_mps, new_velocities_u)
                env.agents[agent_id].set_property_value(c.heading_check_time, check_time + self.check_interval)
                env.heading_turn_counts += 1
                self.log(f'current_step:{cur_step} target_heading:{new_heading} '
                         f'target_altitude_ft:{new_altitude} target_velocities_u_mps:{new_velocities_u}')
        if done:
            self.log(f'agent[{agent_id}] unreached heading. Total Steps={env.current_step}')
            info['heading_turn_counts'] = env.heading_turn_counts
        success = False
        return done, success, info
