# System import
import math
from timeit import default_timer as timer
from typing import Callable, Union, Optional, TypedDict
# External import
import numpy as np
# Custom import 
from configs import DwaConfiguration, CircularRobotSpecification
from pkg_tracker_dwa import utils_geo

from basic_boundary_function.env import GPDFEnv

PathNode = tuple[float, float]

class DebugInfo(TypedDict):
    cost: float
    all_trajectories: list[np.ndarray]
    ok_trajectories: list[np.ndarray]
    ok_costs: list[float]
    step_runtime: float


class TrajectoryTracker:
    """Generate a smooth trajectory tracking based on the reference path and obstacle information.
    
    Attributes:
        config: DWA configuration.
        robot_spec: Robot specification.

    Functions:
        run_step: Run one step of trajectory tracking.

    Comments:
        The solver needs to be built before running the trajectory tracking. \n
        To run the tracker: 
            1. Load motion model and init states; 
            2. Set reference path and trajectory (and states maybe);
            3. Run step.
    """
    def __init__(self, config: DwaConfiguration, robot_specification: CircularRobotSpecification, robot_id:Optional[int]=None, verbose=False):
        self.vb = verbose
        self.robot_id = robot_id if robot_id is not None else 'X'
        self.config = config
        self.robot_spec = robot_specification

        # Common used parameters from config
        self.ts = self.config.ts
        self.ns = self.config.ns
        self.nu = self.config.nu
        self.N_hor = self.config.N_hor

        # Initialization
        self._idle = True
        self._mode: str = 'none'
        self.set_work_mode(mode='work')
        self.vw_range = [robot_specification.lin_vel_min, robot_specification.lin_vel_max,
                        -robot_specification.ang_vel_max, robot_specification.ang_vel_max]

    def load_motion_model(self, motion_model: Callable) -> None:
        """The motion model should be `s'=f(s,a,ts)`.
        - The state needs to be [x, y, theta]
        - The action needs to be [v, w]
        """
        try:
            motion_model(np.array([0,0,0]), np.array([0,0]), 0)
        except Exception as e:
            raise TypeError(f'The motion model doesn\'t satisfy the requirement: {e}')
        self.motion_model = motion_model

    def load_init_states(self, current_state: np.ndarray, goal_state: np.ndarray):
        """Load the initial state and goal state.

        Arguments:
            current_state: Current state of the robot.
            goal_state: Goal state of the robot.

        Attributes:
            state: Current state of the robot.
            final_goal: Goal state of the robot.
            past_states: List of past states of the robot.
            past_actions: List of past actions of the robot.
            cost_timelist: List of cost values of the robot.
            solver_time_timelist: List of solver time of the robot.

        Comments:
            This function resets the `idx_ref_traj/path` to 0 and `idle` to False.
        """
        if (not isinstance(current_state, np.ndarray)) or (not isinstance(goal_state, np.ndarray)):
            raise TypeError(f'State should be numpy.ndarry, got {type(current_state)}/{type(goal_state)}.')
        self.state = current_state
        self.final_goal = goal_state # used to check terminal condition

        self.past_states:list[np.ndarray] = []
        self.past_actions:list[np.ndarray] = []
        self.cost_timelist:list[float] = []
        self.solver_time_timelist:list[float] = []

        self.idx_ref_traj = 0 # for reference trajectory following
        self.idx_ref_path = 0 # for reference path following
        self._idle = False
        self.finishing = False


    def calc_dynamic_window(self, last_v: float, last_w: float):
        """Calculate the dynamic window.
        
        Returns:
            dw: Dynamic window [v_min, v_max, yaw_rate_min, yaw_rate_max]
        """
        vw_now = [min(last_v - self.robot_spec.lin_acc_max*self.ts, 0.1),
                  last_v + self.robot_spec.lin_acc_max*self.ts,
                  last_w - self.robot_spec.ang_acc_max*self.ts,
                  last_w + self.robot_spec.ang_acc_max*self.ts]

        dw = [max(self.vw_range[0], vw_now[0]), min(self.vw_range[1], vw_now[1]),
              max(self.vw_range[2], vw_now[2]), min(self.vw_range[3], vw_now[3])]

        return dw
    
    def pred_trajectory(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Predict the trajectory based on the current state and action.

        Arguments:
            state: Current state of the robot, should be [x, y, angle].
            action: Current action of the robot, should be [v, w].

        Returns:
            trajectory: Predicted trajectory of the robot.
        """
        N_hor = self.config.N_hor
        x = state.copy()
        trajectory = state.copy()
        for i in range(N_hor):
            decay_action = np.array([action[0], action[1]*max(1.0, (N_hor-i)/N_hor)])
            x = self.motion_model(x, decay_action, self.config.ts)
            trajectory = np.vstack((trajectory, x))
        return trajectory[1:]


    def calc_cost_goal_direction(self, trajectory: np.ndarray, goal_state: np.ndarray):
        """Calculate the cost based on the goal direction.
        """
        desired_dir = math.atan2(goal_state[1] - trajectory[0, 1], goal_state[0] - trajectory[0, 0])
        current_dir = math.atan2(trajectory[-1, 1] - trajectory[0, 1], trajectory[-1, 0] - trajectory[0, 0])
        cost_angle = abs(math.atan2(math.sin(desired_dir - current_dir), 
                                    math.cos(desired_dir - current_dir)))
        min_dist = np.min(np.linalg.norm(trajectory[:,:2] - goal_state[:2], axis=1))
        extra_cost = 0.0
        if min_dist > 0.1:
            extra_cost = 10.0
        return cost_angle * self.config.q_goal_dir + extra_cost

    def calc_cost_speed(self, action: np.ndarray):
        return abs(action[0] - self.base_speed) * self.config.q_speed

    def calc_cost_ref_deviation(self, trajectory: np.ndarray, ref_traj: np.ndarray):
        """Calculate the cost based on the reference trajectory."""
        dists = np.linalg.norm(trajectory[:, :2] - ref_traj[:, :2], axis=1)
        cost = np.sum(dists)/self.N_hor * self.config.q_ref_deviation
        if np.max(dists) > 5.0:
            return cost + 10.0
        return cost


    def calc_cost_static_obstacles(self, trajectory: np.ndarray, static_obstacles: list[list[tuple]], thre:float=0.0):
        if len(static_obstacles) == 0:
            return 0.0
        dists_to_obs = []
        for obs in static_obstacles:
            dists = utils_geo.lineseg_dists(trajectory[:,:2], np.array(obs), np.array(obs[1:] + [obs[0]]))
            if np.min(dists) < self.robot_spec.vehicle_width/2:
                return np.inf
            dists_to_obs.append(np.min(dists))
        min_dist = np.min(dists_to_obs)
        if min_dist > thre:
            return 0.0
        return 1.0 / min_dist * self.config.q_stc_obstacle
    
    def calc_cost_dynamic_obstacles(self, trajectory: np.ndarray, dynamic_obstacles: list[tuple], thre:float=0.0):
        set1 = np.expand_dims(trajectory[:, :2], axis=1)
        set2 = np.expand_dims(np.array(dynamic_obstacles), axis=0)
        distances = np.sqrt(np.sum((set1 - set2)**2, axis=-1))
        min_distances = np.min(distances)
        if min_distances > thre + self.robot_spec.social_margin:
            return 0.0
        if min_distances < self.robot_spec.vehicle_width:
            return np.inf
        return 1.0 / min_distances * self.config.q_dyn_obstacle

    def calc_cost_dynamic_obstacles_steps(self, trajectory: np.ndarray, dynamic_obstacles: list[list[tuple]], thre:float=0.0):
        all_step_min_distances = []
        if len(dynamic_obstacles) == 0:
            return 0.0
        for i, obs in enumerate(dynamic_obstacles):
            set1 = np.expand_dims(trajectory[[i], :2], axis=1)
            set2 = np.expand_dims(np.array(obs), axis=0)
            distances = np.sqrt(np.sum((set1 - set2)**2, axis=-1))
            min_distances = np.min(distances) * np.sqrt(i+1)
            if min_distances < self.robot_spec.vehicle_width:
                return np.inf
            all_step_min_distances.append(min_distances)
        
        if np.min(all_step_min_distances) > thre + self.robot_spec.social_margin:
            return 0.0
        return 1.0 / np.min(all_step_min_distances) * self.config.q_dyn_obstacle


    def calc_cost_gpdf(self, traj: np.ndarray, dist_set: np.ndarray, grad_set: Optional[np.ndarray], min_safe_dist:float):
        sa_min = np.radians(120)
        sa_max = np.radians(180)
        dire = np.concatenate((np.cos(traj[:, 2]).reshape(-1, 1), np.sin(traj[:, 2]).reshape(-1, 1)), axis=1)
        cos_angle = np.clip(np.sum(dire * grad_set, axis=1), -1, 1) # clip for numerical stability
        safety_angle = abs(np.arccos(cos_angle))
        safety_angle_pow = safety_angle-sa_min
        safety_angle_pow[safety_angle_pow < 0] = 0
        safety_angle_pow[np.isnan(safety_angle_pow)] = 0
        safety_angle_pow = np.exp(2*safety_angle_pow) - 1

        sa_cost = safety_angle_pow
        sa_cost[sa_cost < sa_min] = sa_min
        sa_cost = (sa_cost - sa_min) / (sa_max - sa_min) #* (np.linspace(1, 0.5, len(sa_cost))) 

        min_dist = np.min(dist_set)
        if min_dist < min_safe_dist:
            return np.inf, np.inf
        if min_dist > 1.0: #min_safe_dist*2 + self.robot_spec.vehicle_margin + self.robot_spec.social_margin:
            return 0.0, 0.0
        dist_cost = 1.0 / min_dist * self.config.q_stc_obstacle

        sa_factor = 1 # len(sa_cost[sa_cost!=0]) * 1.0
        sa_cost = np.sum(sa_cost) / sa_factor if len(sa_cost[sa_cost!=0]) > 0 else 0.0
        return dist_cost, sa_cost


    def calc_traj_cost(self, trajectory: np.ndarray, action: np.ndarray, ref_path: np.ndarray, goal_state: np.ndarray, 
                             static_obstacles: Optional[list[list[tuple]]], dynamic_obstacles: Optional[Union[list[PathNode], list[list[PathNode]]]]) -> float:
        cost_speed = self.calc_cost_speed(action)
        cost_goal_dir = self.calc_cost_goal_direction(trajectory, goal_state)
        cost_ref_deviation = self.calc_cost_ref_deviation(trajectory, ref_path)
        cost_static_obs = 0.0
        cost_dynamic_obs = 0.0
        safe_thre = self.robot_spec.vehicle_width + self.robot_spec.vehicle_margin
        if static_obstacles is not None:
            cost_static_obs = self.calc_cost_static_obstacles(trajectory, static_obstacles)
        if dynamic_obstacles is not None:
            if np.array(dynamic_obstacles).ndim == 2:
                cost_dynamic_obs = self.calc_cost_dynamic_obstacles(trajectory, dynamic_obstacles, thre=safe_thre) # type: ignore
            elif np.array(dynamic_obstacles).ndim == 3:
                cost_dynamic_obs = (
                    self.calc_cost_dynamic_obstacles_steps(trajectory, dynamic_obstacles[1:], thre=safe_thre) + # type: ignore
                    self.calc_cost_dynamic_obstacles(trajectory, dynamic_obstacles[0], thre=safe_thre) # type: ignore
                )
            else:
                cost_dynamic_obs = 0.0

        if cost_static_obs + cost_dynamic_obs < 0.1:
            cost_ref_deviation *= 10
        total_cost = cost_speed + cost_goal_dir + cost_ref_deviation + cost_static_obs + cost_dynamic_obs
        return total_cost
    
    def calc_traj_cost_gpdf(self, trajectory: np.ndarray, action: np.ndarray, 
                                  dist_set:Optional[np.ndarray], grad_set:Optional[np.ndarray]) -> float:
        cost_speed = self.calc_cost_speed(action)
        cost_goal_dir = self.calc_cost_goal_direction(trajectory, self.final_goal)
        cost_ref_deviation = self.calc_cost_ref_deviation(trajectory, self.ref_states)
        cost_static_obs = 0.0
        cost_dynamic_obs = 0.0
        cost_gpdf_obs = 0.0
        if dist_set is not None and len(dist_set[dist_set<np.inf]) > 0:
            dist_cost, sa_cost = self.calc_cost_gpdf(
                trajectory, dist_set, grad_set, 
                min_safe_dist=self.robot_spec.vehicle_width/2)
            cost_gpdf_obs = dist_cost + sa_cost

        total_cost = cost_speed + cost_goal_dir + cost_ref_deviation + cost_gpdf_obs
        # if total_cost < np.inf:
        #     print(f"[{self.__class__.__name__}-{self.robot_id}] Cost: {total_cost:.2f}, Speed: {cost_speed:.2f}, Goal: {cost_goal_dir:.2f}, Ref: {cost_ref_deviation:.2f}, Dist: {dist_cost:.2f}, SA: {sa_cost:.2f}")
        return total_cost


    def set_work_mode(self, mode:str='safe', use_predefined_speed:bool=True):
        """Set the basic work mode (base speed and weight parameters) of the MPC solver.

        Args:
            mode: Be "aligning" (start) or "safe" (20% speed) or "work" (80% speed) or "super" (full speed). Defaults to 'safe'.

        Raises:
            ModuleNotFoundError: If the mode is not supported.

        Attributes:
            base_speed: The reference speed
            tuning_params: Penalty parameters for MPC

        Notes:
            This method will overwrite the base speed.
        """
        if mode == self._mode:
            return

        if self.vb:
            print(f"[{self.__class__.__name__}-{self.robot_id}] Setting work mode to {mode}.")
        self._mode = mode
        if use_predefined_speed:
            if mode == 'aligning':
                self.base_speed = self.robot_spec.lin_vel_max*0.5
            else:
                if mode == 'safe':
                    self.base_speed = self.robot_spec.lin_vel_max*0.2
                elif mode == 'work':
                    self.base_speed = self.robot_spec.lin_vel_max*0.8
                elif mode == 'super':
                    self.base_speed = self.robot_spec.lin_vel_max*1.0
                else:
                    raise ModuleNotFoundError(f'There is no mode called {mode}.')

    def set_current_state(self, current_state: np.ndarray):
        """To synchronize the current state of the robot with the MPC solver."""
        if not isinstance(current_state, np.ndarray):
            raise TypeError(f'State should be numpy.ndarry, got {type(current_state)}.')
        self.state = current_state

    def set_ref_states(self, ref_states: np.ndarray, ref_speed:Optional[float]=None):
        """Set the local reference states for the coming time step.

        Args:
            ref_states: Local (within the horizon) reference states
            ref_speed: The reference speed. If None, use the default speed.
            
        Notes:
            This method will overwrite the base speed.
        """
        self.ref_states = ref_states
        if ref_speed is not None:
            self.base_speed = ref_speed
        else:
            self.set_work_mode(mode='work', use_predefined_speed=True)

    def check_termination_condition(self, external_check=True) -> bool:
        """Check if the robot finishes the trajectory tracking.

        Args:
            external_check: If this is true, the controller will check if it should terminate. Defaults to True.

        Returns:
            _idle: If the robot finishes the trajectory tracking.
        """
        if external_check:
            self.finishing = True
            if np.allclose(self.state[:2], self.final_goal[:2], atol=0.1, rtol=0) and abs(self.past_actions[-1][0]) < 0.1:
                self._idle = True
                if self.vb:
                    print(f"[{self.__class__.__name__}-{self.robot_id}] Trajectory tracking finished.")
        return self._idle


    def run_step(self,
                 enable_gpdf:bool=False,
                 static_obstacles:Optional[list[list[PathNode]]]=None,
                 dyn_obstacle_list:Optional[Union[list[tuple], list[list[tuple]]]]=None,
                 other_robot_states:Optional[list]=None,
                 gpdf_env:Optional[GPDFEnv]=None,
                 last_action:Optional[np.ndarray]=None):
        """Run the trajectory planner for one step. If enable_gpdf is True, use GPDF to calculate the cost.

        Returns:
            best_u: The best action to take.
            best_trajectory: The predicted trajectory.
            ref_states: The reference states.
            debug_info: Debug information.
        """
        if enable_gpdf:
            if gpdf_env is None:
                raise ValueError("The GPDF environment should be provided.")
            return self.run_step_gpdf(gpdf_env, last_action)
        else:
            return self.run_step_regular(static_obstacles, dyn_obstacle_list, other_robot_states, last_action)

    def run_step_regular(self, 
                         static_obstacles: Optional[list[list[PathNode]]], 
                         dyn_obstacle_list: Optional[Union[list[tuple], list[list[tuple]]]], 
                         other_robot_states:Optional[list]=None,
                         last_action:Optional[np.ndarray]=None):
        """Run the trajectory planner for one step.

        Returns:
            best_u: The best action to take.
            best_trajectory: The predicted trajectory.
            ref_states: The reference states.
            debug_info: Debug information.
        """

        ### Correct the reference speed ###
        dist_to_goal = math.hypot(self.state[0]-self.final_goal[0], self.state[1]-self.final_goal[1]) # change ref speed if final goal close
        if dist_to_goal < self.base_speed*self.N_hor*self.ts:
            self.base_speed = min(2 * dist_to_goal / self.N_hor / self.ts, self.robot_spec.lin_vel_max)

        min_cost = float("inf")
        x_init = self.state.copy()
        best_u = np.zeros(self.nu)
        best_trajectory = self.state.reshape(1, -1)
        if last_action is not None:
            last_u = last_action
        else:
            last_u = self.past_actions[-1] if len(self.past_actions) else np.zeros(self.nu)
        all_trajectories = []
        ok_trajectories = []
        ok_costs = []

        start_time = timer()
        ### Get dynamic window ###
        dw = self.calc_dynamic_window(last_u[0], last_u[1])

        ### Get reference states ###
        for v in np.arange(dw[0], dw[1]+self.config.vel_resolution, self.config.vel_resolution):
            for w in np.arange(dw[2], dw[3]+self.config.ang_resolution, self.config.ang_resolution):
                u = np.array([v, w])
                trajectory = self.pred_trajectory(x_init, u)
                all_trajectories.append(trajectory)
                cost = self.calc_traj_cost(
                    trajectory, 
                    u, 
                    np.array(self.ref_states), 
                    self.final_goal, 
                    static_obstacles, 
                    dyn_obstacle_list,
                )
                if cost < np.inf:
                    ok_trajectories.append(trajectory)
                    ok_costs.append(cost)
                if min_cost > cost:
                    min_cost = cost
                    best_u = u
                    best_trajectory = trajectory
                    if abs(best_u[0]) < self.config.stuck_threshold:
                        best_u[1] = -self.robot_spec.ang_vel_max
        solver_time = timer() - start_time

        self.state = best_trajectory[0, :]
        self.past_states.append(self.state)
        self.past_actions += [best_u]
        self.cost_timelist.append(cost)
        self.solver_time_timelist.append(solver_time)
        debug_info = DebugInfo(cost=min_cost, 
                               all_trajectories=all_trajectories, 
                               ok_trajectories=ok_trajectories, 
                               ok_costs=ok_costs, 
                               step_runtime=solver_time)

        return best_u, best_trajectory, self.ref_states, debug_info

    def run_step_gpdf(self, 
                      gpdf_env: GPDFEnv,
                      last_action:Optional[np.ndarray]=None):
        """Run the trajectory planner for one step.

        Returns:
            best_u: The best action to take.
            best_trajectory: The predicted trajectory.
            ref_states: The reference states.
            debug_info: Debug information.
        """
        ### Correct the reference speed ###
        dist_to_goal = math.hypot(self.state[0]-self.final_goal[0], self.state[1]-self.final_goal[1]) # change ref speed if final goal close
        dist_to_end_ref = math.hypot(self.state[0]-self.ref_states[-1, 0], self.state[1]-self.ref_states[-1, 1])
        dist_to_check = max(dist_to_goal, dist_to_end_ref)
        if dist_to_check < self.base_speed*self.N_hor*self.ts:
            new_speed = min(dist_to_check / self.N_hor / self.ts + 0.08, self.robot_spec.lin_vel_max)
            self.set_work_mode(mode='work', use_predefined_speed=True)
            if new_speed < self.base_speed:
                self.base_speed = new_speed

        min_cost = float("inf")
        x_init = self.state.copy()
        best_u = np.zeros(self.nu)
        best_trajectory = self.state.reshape(1, -1)
        if last_action is not None:
            last_u = last_action
        else:
            last_u = self.past_actions[-1] if len(self.past_actions) else np.zeros(self.nu)
        all_trajectories = []
        ok_trajectories = []
        ok_costs = []

        start_time = timer()
        ### Get dynamic window ###
        dw = self.calc_dynamic_window(last_u[0], last_u[1])

        ### Get reference states ###
        v_list = np.arange(dw[0], dw[1]+self.config.vel_resolution, self.config.vel_resolution)
        w_list = np.arange(dw[2], dw[3]+self.config.vel_resolution, self.config.ang_resolution)
        all_u = np.array(np.meshgrid(v_list, w_list)).T.reshape(-1, 2) # N_u x 2
        for i in range(len(all_u)):
            u = all_u[i]
            trajectory = self.pred_trajectory(x_init, u)
            all_trajectories.append(trajectory)
        all_trajectories_np = np.concatenate(all_trajectories, axis=0) # (N_hor+1)*N_traj x 3
        dist_set_all, grad_set_all = gpdf_env.h_grad_vector(np.asarray(all_trajectories_np)[:, :2], exclude_index=f'other_robots_{self.robot_id}')
        for i, trajectory in enumerate(all_trajectories):
            u = all_u[i]
            dist_set = dist_set_all[i*len(trajectory):(i+1)*len(trajectory)]
            grad_set = grad_set_all[i*len(trajectory):(i+1)*len(trajectory)]
            cost = self.calc_traj_cost_gpdf(trajectory, u, dist_set, grad_set)
            if cost < np.inf:
                ok_trajectories.append(trajectory)
                ok_costs.append(cost)
            if min_cost > cost:
                min_cost = cost
                best_u = u
                best_trajectory = trajectory
                if abs(best_u[0]) < self.config.stuck_threshold:
                    best_u[1] = -self.robot_spec.ang_vel_max
        solver_time = timer() - start_time

        self.state = best_trajectory[0, :]
        self.past_states.append(self.state)
        self.past_actions += [best_u]
        self.cost_timelist.append(min_cost)
        self.solver_time_timelist.append(solver_time)
        debug_info = DebugInfo(cost=min_cost, 
                               all_trajectories=all_trajectories, 
                               ok_trajectories=ok_trajectories, 
                               ok_costs=ok_costs, 
                               step_runtime=solver_time)

        return best_u, best_trajectory, self.ref_states, debug_info


class TrajectoryPlanner(TrajectoryTracker):
    def __init__(self, config: DwaConfiguration, robot_specification: CircularRobotSpecification, verbose=False):
        super().__init__(config, robot_specification, robot_id=None, verbose=verbose)
    
