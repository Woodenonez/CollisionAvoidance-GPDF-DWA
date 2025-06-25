import math
import time
import pathlib
import warnings
from timeit import default_timer as timer
from typing import Optional, Union

import numpy as np
import matplotlib.pyplot as plt # type: ignore
from matplotlib.axes import Axes # type: ignore

from configs import MpcConfiguration
from configs import DwaConfiguration
from configs import TebConfiguration
from configs import CircularRobotSpecification
from configs import PedestrianSpecification

from basic_boundary_function.env import GPDFEnv
from basic_sdf.esdf import EuclideanSDF

from pkg_motion_plan.global_path_coordinate import GlobalPathCoordinator
from pkg_motion_plan.local_traj_plan import LocalTrajPlanner
from pkg_robot.robot import RobotManager
from pkg_moving_object.moving_object import HumanObject, RobotObject

from pkg_tracker_mpc.trajectory_tracker import TrajectoryTracker as TrajectoryTrackerMPC
from pkg_tracker_dwa.trajectory_tracker import TrajectoryTracker as TrajectoryTrackerDWA
from pkg_tracker_dwa.trajectory_tracker import TrajectoryPlanner as TrajectoryPlannerDWA
from pkg_planner_teb.trajectory_planner import TrajectoryPlanner as TrajectoryPlannerTEB

from visualizer.object import CircularObjectVisualizer
from visualizer.mpc_plot import MpcPlotInLoop # type: ignore

from evaluation import Evaluator
from pre_maps import generate_map

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


class MainBase:
    color_list = ['#2878b5', '#9ac9db', '#f8ac8c', '#c82423', '#bb9727', 
                  '#54b345', '#32b897', '#05b9e2', '#8983bf', '#c76da2',
                  '#f8ac8c', '#c82423', '#bb9727', '#54b345', '#32b897',]
    
    def __init__(self, scenario_index: tuple[int, int, int], evaluation=False, map_only=False, save_video_name:Optional[str]=None, verbose=True):
        (self.robot_starts, self.robot_goals, 
         self.human_starts, self.human_paths, 
         self._map_boundary, self._map_obstacle_list) = generate_map(*scenario_index) 

        self.eval = evaluation
        self.map_only = map_only
        self.viz = not evaluation
        self.save_video_name = save_video_name
        self.vb = verbose

    def _load_config(self, config_planner_path, config_tracker_path, config_robot_path, config_human_path):
        if self.vb:
            print(f'[{self.__class__.__name__}] Loading configuration...')
        if self.tracker_type == 'mpc':
            self.config_tracker = MpcConfiguration.from_yaml(config_tracker_path)
        elif self.tracker_type in ['dwa', 'dwa-gpdf', 'dwa-esdf', 'dwa-pred']:
            self.config_tracker = DwaConfiguration.from_yaml(config_tracker_path)
            
        if self.planner_type == 'teb':
            self.config_planner = TebConfiguration.from_yaml(config_planner_path)
        elif self.planner_type in ['dwa', 'dwa-gpdf', 'dwa-esdf', 'dwa-pred']:
            self.config_planner = DwaConfiguration.from_yaml(config_planner_path)
        self.config_robot = CircularRobotSpecification.from_yaml(config_robot_path)
        self.config_human = PedestrianSpecification.from_yaml(config_human_path)

    def _prepare_global_coordinator(self):
        if self.vb:
            print(f'[{self.__class__.__name__}] Preparing global coordinator...')

        inflation_margin     = (self.config_robot.vehicle_width/2)
        inflation_margin_mpc = (self.config_robot.vehicle_width/2+self.config_robot.vehicle_margin+0.1)

        self.robot_ids = list(range(len(self.robot_starts)))
        robot_schedule_list = []
        for r_id, r_start, r_goals in zip(self.robot_ids, self.robot_starts, self.robot_goals):
            robot_schedule_list.append((r_id, tuple(r_start), r_goals))
        
        self.gpc = GlobalPathCoordinator.simple_schedule(robot_schedule_list)
        self.gpc_mpc = GlobalPathCoordinator.simple_schedule(robot_schedule_list)
        self.gpc.load_map(self._map_boundary, self._map_obstacle_list, inflation_margin=inflation_margin)
        self.gpc_mpc.load_map(self._map_boundary, self._map_obstacle_list, inflation_margin=inflation_margin_mpc)

        self.gpdf_env = GPDFEnv()
        if self._map_obstacle_list:
            self.gpdf_env.add_gpdfs_after_interp(
                list(range(len(self._map_obstacle_list))), 
                pc_coords_list=[np.asarray(x) for x in self._map_obstacle_list],
                interp_res=0.1
            )

        self.esdf_env = EuclideanSDF()
        if self._map_obstacle_list:
            self.esdf_env.add_obstacles_after_interp(
                list(range(len(self._map_obstacle_list))), 
                obstacle_points_list=[np.asarray(x) for x in self._map_obstacle_list],
                interp_res=0.1
            )

        self.static_obstacles = self.gpc.inflated_map.obstacle_coords_list
        self.static_obstacles_mpc = self.gpc_mpc.inflated_map.obstacle_coords_list

    def _prepare_robot_manager(self):
        if self.vb:
            print(f'[{self.__class__.__name__}] Preparing robots...')
        self.robot_manager = RobotManager()

        for rid, rstart in zip(self.robot_ids, self.robot_starts):
            robot = self.robot_manager.create_robot(self.config_robot, motion_model=None, id_=rid)
            robot.set_state(rstart)
            planner = LocalTrajPlanner(self.config_tracker.ts, self.config_tracker.N_hor, self.config_robot.lin_vel_max, verbose=False)
            planner.load_map(self.gpc.inflated_map.boundary_coords, self.gpc.inflated_map.obstacle_coords_list)
            if self.tracker_type == 'mpc':
                controller = TrajectoryTrackerMPC(self.config_tracker, self.config_robot, robot_id=rid, verbose=False)
                controller.load_motion_model(robot.motion_model)
                # controller.set_monitor(monitor_on=False)
            elif 'dwa' in self.tracker_type:
                controller = TrajectoryTrackerDWA(self.config_tracker, self.config_robot, robot_id=rid, verbose=False)
                controller.load_motion_model(robot.motion_model)
            visualizer = CircularObjectVisualizer(self.config_robot.vehicle_width/2, indicate_angle=True)
            self.robot_manager.add_robot(robot, controller, planner, visualizer)

    def _prepare_evaluation(self, repeat:int=1):
        self.evaluator = Evaluator(self.robot_ids,
                                   self.config_robot.vehicle_width/2,
                                   self.config_human.human_width,
                                   repeat=repeat)

    def _prepare_pedestrians(self, pedestrian_model:Optional[str]=None):
        """
        Args:
            pedestrian_model: Can be `None` (non-interactive), 'sf' (social force).
        """
        if self.vb:
            print(f'[{self.__class__.__name__}] Preparing agents...')
        self.pedestrian_model = pedestrian_model
        self.humans = [HumanObject(h_s, self.config_robot.ts, self.config_human.human_width, self.config_human.human_stagger) for h_s in self.human_starts]
        human_paths_coords = self.human_paths
        for i, human in enumerate(self.humans):
            human.set_path(human_paths_coords[i])
            if pedestrian_model == 'sf':
                human.set_social_repulsion(max_distance=5.0, max_angle=math.pi/4, max_force=1.0, opponent_type=RobotObject)
            elif pedestrian_model == 'minisf':
                human.set_social_repulsion(max_distance=3.0, max_angle=math.pi/4, max_force=0.5, opponent_type=RobotObject)
        self.human_visualizers = [CircularObjectVisualizer(self.config_human.human_width, indicate_angle=False) for _ in self.humans]

    @property
    def planner_type(self):
        return self._planner_type

    @property
    def tracker_type(self):
        return self._tracker_type

    def load(self, 
             planner_type: Optional[str], config_planner_path: str, 
             tracker_type: Optional[str], config_tracker_path: str, 
             config_robot_path: str, config_human_path: str):
        """
        Args:
            planner_type: The type of the planner, 'none' or 'teb'.
            tracker_type: The type of the trajectory tracker, 'mpc' or 'dwa'.
        """
        self._planner_type = planner_type
        self._tracker_type = tracker_type
        self._load_config(config_planner_path, config_tracker_path, config_robot_path, config_human_path)
        print(f'[{self.__class__.__name__}] Loading complete!')

    def prepare(self, pedestrian_model:Optional[str]=None):
        """Prepare the simulation (coordinator, robots, motion predictor, pedestrians)

        Args:
            motion_predictor_type: The type of the motion predictor, 'cvm', 'sgan', 'nll', 'enll', 'bce', 'kld'.
            ref_image_path: The reference image to plot as the background.
            model_suffix: The suffix of the model to load, used to quickly load different models.
            pedestrian_model: The model for the pedestrian, can be `None` (non-interactive), 'sf' (social force).
        """
        prt_process = 'Preparing: Coordinator'
        process_timer = timer()
        print(f'[{self.__class__.__name__}] {prt_process}', end=' ')
        self._prepare_global_coordinator()

        prt_process += f' {round(timer()-process_timer, 3)}s | Manager'
        process_timer = timer()
        print(f'\r[{self.__class__.__name__}] {prt_process}', end=' ')
        self._prepare_robot_manager()
        self.generic_planner:Optional[Union[TrajectoryPlannerTEB, TrajectoryPlannerDWA]] = None
        if self.planner_type == 'teb':
            self.generic_planner = TrajectoryPlannerTEB(self.config_planner, self.config_robot, safe_factor=3.0, safe_margin=0.1, verbose=self.vb)
        elif self.planner_type in ['dwa', 'dwa-gpdf', 'dwa-esdf']:
            self.generic_planner = TrajectoryPlannerDWA(self.config_planner, self.config_robot)
            self.generic_planner.load_motion_model(self.robot_manager.get_all_robots()[0].motion_model)

        prt_process += f' {round(timer()-process_timer, 3)}s | Predictor'
        process_timer = timer()
        print(f'\r[{self.__class__.__name__}] {prt_process}', end=' ')

        for rid, rstart in zip(self.robot_ids, self.robot_starts):
            path_coords, path_times = self.gpc.get_robot_schedule(rid)
            self.robot_manager.add_schedule(rid, rstart, path_coords, path_times)

        prt_process += f' {round(timer()-process_timer, 3)}s | Pedestrian'
        process_timer = timer()
        print(f'\r[{self.__class__.__name__}] {prt_process}', end=' ')
        self._prepare_pedestrians(pedestrian_model)

        prt_process += f' {round(timer()-process_timer, 3)}s | Visualizer'
        process_timer = timer()
        print(f'\r[{self.__class__.__name__}] {prt_process}', end=' ')
        if self.viz:
            if self.save_video_name is not None:
                self.main_plotter = MpcPlotInLoop(self.config_robot, map_only=self.map_only, save_to_path=self.save_video_name, save_params={'skip_frame': 0})
            else:
                self.main_plotter = MpcPlotInLoop(self.config_robot, map_only=self.map_only)
            # graph_manager = self.gpc.current_graph
            graph_manager = None
            self.main_plotter.plot_in_loop_pre(self.gpc.current_map, self.gpc.inflated_map, graph_manager)
            
            for rid in self.robot_ids:
                robot = self.robot_manager.get_robot(rid)
                planner = self.robot_manager.get_planner(rid)
                controller = self.robot_manager.get_controller(rid)
                visualizer = self.robot_manager.get_visualizer(rid)
                self.main_plotter.add_object_to_pre(rid,
                                                    planner.ref_traj,
                                                    controller.state,
                                                    controller.final_goal,
                                                    color=self.color_list[rid])
                visualizer.plot(self.main_plotter.map_ax, *robot.state, object_color=self.color_list[rid])

            for _, (human, human_vis) in enumerate(zip(self.humans, self.human_visualizers)):
                human_vis.plot(self.main_plotter.map_ax, x=human.state[0], y=human.state[1], object_color='m')

        prt_process += f' {round(timer()-process_timer, 3)}s'
        print(f'\r[{self.__class__.__name__}] {prt_process}')

        print(f'[{self.__class__.__name__}] Preparation complete!')

    def reset(self):
        if self.viz:
            self.main_plotter.close()
            plt.close('all')

        self._prepare_robot_manager()
        for rid, rstart in zip(self.robot_ids, self.robot_starts):
            path_coords, path_times = self.gpc.get_robot_schedule(rid)
            self.robot_manager.add_schedule(rid, rstart, path_coords, path_times)
        self._prepare_pedestrians(pedestrian_model=self.pedestrian_model)

        if self.viz:
            self.main_plotter = MpcPlotInLoop(self.config_robot, map_only=self.map_only)
            # graph_manager = self.gpc.current_graph
            graph_manager = None
            self.main_plotter.plot_in_loop_pre(self.gpc.current_map, self.gpc.inflated_map, graph_manager)
            
            for rid in self.robot_ids:
                robot = self.robot_manager.get_robot(rid)
                planner = self.robot_manager.get_planner(rid)
                controller = self.robot_manager.get_controller(rid)
                visualizer = self.robot_manager.get_visualizer(rid)
                self.main_plotter.add_object_to_pre(rid,
                                                    planner.ref_traj,
                                                    controller.state,
                                                    controller.final_goal,
                                                    color=self.color_list[rid])
                visualizer.plot(self.main_plotter.map_ax, *robot.state, object_color=self.color_list[rid])

            for i, (human, human_vis) in enumerate(zip(self.humans, self.human_visualizers)):
                human_vis.plot(self.main_plotter.map_ax, *human.state, object_color='r')

        print(f'[{self.__class__.__name__}] Reset the simulation!')


    def run_one_step(self, current_time: float, num_repeat: int, extra_debug_panel:Optional[list[Axes]]=None, auto_run:bool=True):
        """Run one step of the simulation.

        Args:
            current_time: The current time (real time but not time step).
            num_repeat: The index of the current repeat.
            extra_debug_panel: _description_. Defaults to None.

        Returns:
            clusters_list: A list (n=horizon) of clusters, each cluster is a list of points.
            mu_list_list: A list of means of the clusters.
            std_list_list: A list of standard deviations of the clusters.
            conf_list_list: A list of confidence of the clusters.
        """
        kt = current_time / self.config_tracker.ts

        ### Human step
        sf_viz_list = [] # type: ignore
        for i, human in enumerate(self.humans):
            agents = [self.robot_manager.get_robot(rid).robot_object for rid in self.robot_ids]
            social_force, _, attenuation_factor = human.get_social_repulsion(agents) # will work only if the human is set to social force model
            action = human.run_step(self.config_human.human_vel_max, social_force=social_force, attenuation_factor=attenuation_factor)

            if self.viz:
                self.human_visualizers[i].update(*human.state)
        
        ### Run Tracker
        dynamic_obstacles = [human.state[:2].tolist() for human in self.humans] # [[x1, y1], [x2, y2], ...]

        for rid in self.robot_ids:
            if self.evaluator.eval_results[rid]["complete_results"][num_repeat]:
                continue # skip if this robot has completed

            robot = self.robot_manager.get_robot(rid)
            planner = self.robot_manager.get_planner(rid)
            controller = self.robot_manager.get_controller(rid)
            visualizer = self.robot_manager.get_visualizer(rid)

            if (len(self.robot_manager) > 1) and self.tracker_type in ['mpc', 'dwa', 'dwa-gpdf', 'dwa-esdf', 'dwa-pred']:
                if self.tracker_type == 'mpc':
                    default_value = -10.0
                    n_others = self.config_tracker.Nother
                elif 'dwa' in self.tracker_type:
                    default_value = np.inf
                    n_others = len(self.robot_manager)-1
                other_robot_states = self.robot_manager.get_other_robot_states(
                    rid, 
                    n_state=self.config_tracker.ns, 
                    n_horizon=self.config_tracker.N_hor, 
                    n_others=n_others,
                    sep=('dwa' in self.tracker_type),
                    default=default_value
                )
            else:
                other_robot_states = None

            if ('dwa' in self.tracker_type) and (other_robot_states is not None):
                assert isinstance(other_robot_states, dict)
                other_robot_states_pred = []
                for other_id, other_states in other_robot_states.items():
                    other_robot_states_pred.append(other_states)
                    other_robot_states_np = np.asarray([x for x in other_states if x != default_value]).reshape(-1, 3)
                    other_robot_obstacle = np.unique(other_robot_states_np[:, :2], axis=0)
                    if other_robot_obstacle.shape[0] == 1:
                        other_robot_obstacle = np.concatenate(
                            (
                                other_robot_obstacle, 
                                other_robot_obstacle + np.array([0.0, 0.1]),
                                other_robot_obstacle + np.array([0.1, 0.0]),
                            ),
                            axis=0
                        )
                    self.gpdf_env.add_gpdf(index=f'other_robots_{other_id}', pc_coords=other_robot_obstacle)
                try:
                    other_robot_states_pred_np = np.array(other_robot_states_pred)[:, :, None].reshape(len(other_robot_states), -1, 3)
                    other_robot_states_pred_np = np.transpose(other_robot_states_pred_np, axes=[1, 0, 2])[:, :, :2]
                except ValueError:
                    other_robot_states_pred_np = np.array([robot.state[:2] for robot in self.robot_manager.get_all_robots() if robot.id_ != rid])

            controller.set_current_state(robot.state)
            if controller.finishing:
                ref_states, ref_speed, *_ = planner.get_local_ref(kt*self.config_tracker.ts, (float(robot.state[0]), float(robot.state[1])))
            else:
                ref_states, ref_speed, *_ = planner.get_local_ref(kt*self.config_tracker.ts, (float(robot.state[0]), float(robot.state[1])), external_ref_speed=controller.base_speed)
            if self.planner_type == 'teb':
                assert isinstance(self.generic_planner, TrajectoryPlannerTEB)
                self.generic_planner.set_ref_states(robot.state, ref_states, ref_speed)
                ref_states, _ = self.generic_planner.run_step(obstacles=dynamic_obstacles,
                                                              obstacle_radius=self.config_human.human_width*1.5)
            elif self.planner_type in ['dwa', 'dwa-gpdf', 'dwa-esdf', 'dwa-pred']:
                assert isinstance(self.generic_planner, TrajectoryPlannerDWA)
                self.generic_planner.load_init_states(robot.state, goal_state=self.robot_manager.get_goal_state(rid))
                self.generic_planner.set_ref_states(ref_states, ref_speed=ref_speed)

                if self.planner_type == 'dwa-pred':
                    dyn_obstacle_list = other_robot_states_pred_np.tolist()
                else:
                    dyn_obstacle_list = [robot.state[:2] for robot in self.robot_manager.get_all_robots() if robot.id_ != rid]
                
                _, ref_states, _, _ = self.generic_planner.run_step(
                    enable_gdf=(self.planner_type in ['dwa-gpdf', 'dwa-esdf']),
                    static_obstacles=self.static_obstacles,
                    dyn_obstacle_list=dyn_obstacle_list,
                    gpdf_env=self.gpdf_env if (self.planner_type=='dwa-gpdf') else None,
                    esdf_env=self.esdf_env if (self.planner_type=='dwa-esdf') else None,
                    last_action=controller.past_actions[-1] if len(controller.past_actions) > 0 else None,
                )


            controller.set_ref_states(ref_states, ref_speed=ref_speed)

            start_tracker_time = timer()
            if self.tracker_type == 'mpc':
                dynamic_obstacles_with_distance = [list(human.state[:2])+[self.config_human.human_width] for human in self.humans] # [[x1, y1, r], [x2, y2, r], ...]
                assert isinstance(controller, TrajectoryTrackerMPC)
                actions, pred_states, current_refs, debug_info = controller.run_step(
                    static_obstacles=self.static_obstacles_mpc,
                    dyn_obstacle_list=dynamic_obstacles_with_distance,
                    other_robot_states=other_robot_states,
                    map_updated=True)
                action = actions[-1]
            elif 'dwa' in self.tracker_type:
                assert isinstance(controller, TrajectoryTrackerDWA)
                if self.tracker_type == 'dwa-pred':
                    dyn_obstacle_list=other_robot_states_pred_np.tolist()
                else:
                    dyn_obstacle_list=[robot.state[:2] for robot in self.robot_manager.get_all_robots() if robot.id_ != rid]

                action, pred_states, current_refs, debug_info = controller.run_step(
                    enable_gdf=(self.tracker_type in ['dwa-gpdf', 'dwa-esdf']),
                    static_obstacles=self.static_obstacles,
                    dyn_obstacle_list=dyn_obstacle_list,
                    gpdf_env=self.gpdf_env if (self.tracker_type=='dwa-gpdf') else None,
                    esdf_env=self.esdf_env if (self.tracker_type=='dwa-esdf') else None,
                    last_action=controller.past_actions[-1] if len(controller.past_actions) > 0 else None,
                )
                
            solve_tracker_time = timer() - start_tracker_time
            self.evaluator.append_tracker_solve_time(rid, solve_tracker_time)

            self.robot_manager.set_pred_states(rid, np.asarray(pred_states))

            assert isinstance(action, np.ndarray)

            ### Robot step
            if self.tracker_type == 'mpc':
                if (action[0]<self.config_robot.lin_vel_min*0.9 or action[0]==1.5 or debug_info['cost']>1e3):
                    controller.restart_solver()
            robot.step(action)

            if controller.check_termination_condition(external_check=planner.idle):
                self.evaluator.check_completion(rid, num_repeat, True)

            ### Check collisions
            other_states = [r.state for r in self.robot_manager.get_all_robots() if r.id_ != rid]
            have_collision = self.evaluator.check_collision(rid, num_repeat, robot.state, other_states, self.static_obstacles, dynamic_obstacles)
            # if have_collision:
            #     print(f"COLLISION COST: {debug_info['cost']}")
            #     input("Detect collision! Press Enter to continue...")
            if self.eval and dynamic_obstacles:
                self.evaluator.calc_minimal_dynamic_obstacle_distance(rid, num_repeat, robot.state, dynamic_obstacles)

            if self.viz:
                self.main_plotter.update_plot(rid, kt, action, robot.state, min(debug_info['cost'], 10000), np.asarray(pred_states), current_refs)
                visualizer.update(*robot.state)

            if self.vb:
                assert action is not None, f"Action is None for robot {rid}."
                prt_action = f'Actions:({round(action[0], 4)}, {round(action[1], 4)});'
                prt_state  = f'Robot state: R/T {[round(x,4) for x in robot.state]}/{[round(x,4) for x in controller.state]};'
                prt_cost   = f"Cost:{round(debug_info['cost'],4)}."
                print(f"[{self.__class__.__name__}] Time:{current_time:.2f} | Robot {rid} |", prt_action, prt_state, prt_cost)
            else:
                print(f"[{self.__class__.__name__}] Time:{current_time:.2f}", end='\r')

        if self.viz:
            tracker_viz = []
            if 'dwa' in self.tracker_type:
                all_trajs = debug_info['all_trajectories']
                ok_trajs = debug_info['ok_trajectories']
                ok_cost = debug_info['ok_costs']
                for tr in all_trajs:
                    viz = self.main_plotter.map_ax.plot(tr[:, 0], tr[:, 1], 'gray', alpha=0.3)[0]
                    tracker_viz.append(viz)
                for tr, c in zip(ok_trajs, ok_cost):
                    c_normalized = (c - min(ok_cost)) / (max(ok_cost) - min(ok_cost) + 1e-3)
                    alpha = (1 - c_normalized) * 0.5
                    viz = self.main_plotter.map_ax.plot(tr[:, 0], tr[:, 1], 'b', alpha=alpha)[0]
                    # viz_text = self.main_plotter.map_ax.text(tr[-1][0], tr[-1][1], f'{round(c,2)}', fontsize=8, color='m')
                    tracker_viz.append(viz)
                    # tracker_viz.append(viz_text)
                viz = self.main_plotter.map_ax.plot(np.asarray(pred_states)[:, 0], np.asarray(pred_states)[:, 1], 'r', linewidth=3)[0]
                tracker_viz.append(viz)

            ctr, ctrf, map_quiver = None, None, None
            gdf_map_resolution = (100, 100)
            if self.tracker_type == 'dwa-gpdf' or self.planner_type == 'dwa-gpdf':
                ctr, ctrf, map_quiver, _ = self.gpdf_env.plot_env(
                    self.main_plotter.map_ax,
                    x_range=(min([x[0] for x in self._map_boundary]), max([x[0] for x in self._map_boundary])),
                    y_range=(min([x[1] for x in self._map_boundary]), max([x[1] for x in self._map_boundary])),
                    map_resolution=gdf_map_resolution,
                    color='k',
                    plot_grad_dir=False,
                    obstacle_idx=-1,
                    show_grad=True,
                    exclude_index=f'other_robots_{rid}',
                )
            elif self.tracker_type == 'dwa-esdf' or self.planner_type == 'dwa-esdf':
                ctr, ctrf, map_quiver, _ = self.esdf_env.plot_env(
                    self.main_plotter.map_ax,
                    x_range=(min([x[0] for x in self._map_boundary]), max([x[0] for x in self._map_boundary])),
                    y_range=(min([x[1] for x in self._map_boundary]), max([x[1] for x in self._map_boundary])),
                    map_resolution=gdf_map_resolution,
                    color='k',
                    plot_grad_dir=False,
                    show_grad=True,
                )
            boundary_np = np.asarray(self._map_boundary)
            # zoom_in = [np.min(boundary_np[:, 0])-0.1, np.max(boundary_np[:, 0])+0.1, np.min(boundary_np[:, 1])-0.1, np.max(boundary_np[:, 1])+0.1]
            # zoom_in = [2, 10, 1, 7] # for (1, 3, 1) concept plotting
            # zoom_in = [-1, 11, -1, 11] # for (3, 1, 1) 
            self.main_plotter.plot_in_loop(
                dyn_obstacle_list=None, 
                time=current_time, 
                autorun=auto_run, 
                # zoom_in=zoom_in, 
                # save_path=f'Demo/{int(current_time/self.config_tracker.ts)}.png',
                temp_plots=sf_viz_list + [ctr, ctrf, map_quiver] + tracker_viz,
                # temp_objects=debug_info['closest_obstacle_list']
            )
            for i, human in enumerate(self.humans):
                self.human_visualizers[i].update(*human.state)

    def run_once(self, repeat:int=1, time_step_out:Optional[float]=None, extra_debug_panel:Optional[list[Axes]]=None, auto_run:bool=True):
        """Run the simulation once for a given number of repeats.

        Args:
            repeat: The number of repeats. Defaults to 1.
            time_step_out: The time to stop the simulation if not None. Defaults to None.
            extra_debug_panel: List of Axes for extra debug panel if not None. Defaults to None.
            auto_run: If True, the plot will be updated automatically (without plt.waitforbuttonpress). Defaults to True.
        """
        self._prepare_evaluation(repeat=repeat)
        if repeat > 1 and self.viz:
            warnings.warn("Try to avoid visualization when repeat > 1.")

        first_run = True
        for i in range(repeat):
            print(f'[{self.__class__.__name__}] Repeat {i+1}/{repeat}:')

            if first_run:
                assert i==0, "The first run should be the first repeat (index 0)."
                time.sleep(1.0)
                first_run = False

            current_time = 0.0
            total_complete = False
            any_collision = False
            while (not total_complete) and (not any_collision):
                self.run_one_step(current_time=current_time,
                                  num_repeat=i, 
                                  extra_debug_panel=extra_debug_panel,
                                  auto_run=auto_run)
                total_complete = all([self.evaluator.eval_results[rid]["complete_results"][i] for rid in self.robot_ids])
                any_collision = any([self.evaluator.eval_results[rid]["collision_results"][i] for rid in self.robot_ids])
                
                if (time_step_out is not None) and (current_time+1e-6 >= time_step_out):
                    break
                current_time += self.config_tracker.ts
                if self.viz:
                    time.sleep(0.2)
                else:
                    time.sleep(0.01)

            if total_complete and self.eval:
                for rid in self.robot_ids:
                    self.evaluator.calc_action_smoothness(rid, i, self.robot_manager.get_controller(rid).past_actions)
                    self.evaluator.calc_minimal_obstacle_distance(rid, i, self.robot_manager.get_controller(rid).past_states, self.static_obstacles)
                    self.evaluator.calc_deviation_distance(rid, i, ref_traj=self.robot_manager.get_planner(rid).ref_traj, actual_traj=self.robot_manager.get_controller(rid).past_states)

            if not self.vb:
                print() # print a new line

            print(f'[{self.__class__.__name__}] Repeat {i+1} finished. Any collision: {any_collision}. All complete: {total_complete}. Timeout: {(not total_complete) and (not any_collision)}.')

            if self.viz:
                if any_collision and self.save_video_name is None:
                    input(f"[{self.__class__.__name__}] Collision detected. Press Enter to continue...")
                # self.main_plotter.show()

            if i < repeat-1:

                self.reset()

        if self.viz: 
            if auto_run and self.save_video_name is None:
                input("Press Enter to continue...")
            self.main_plotter.close()
            plt.close('all')


    def report(self, save_dir:Optional[str]=None):
        for rid in self.robot_ids:
            save_path = None     
            if save_dir is not None:
                save_path = f'{save_dir}/robot_{rid}_results.json'
            self.evaluator.report(rid, full_report=self.eval, save_path=save_path)


if __name__ == '__main__':
    import os
    """All scenarios in evaluation:
    Single robot:
        - (1, 1, 3) - Large rectangular obstacle
        - (1, 2, 2) - Two large stagger obstacles
        - (1, 3, 1) - Large U-shaped obstacle
        - (2, 1, 2) - Sharp corner
        - (2, 1, 3) - U-turn
    Multi-robot:np.linspace(0, len(sa_cost), len(sa_cost))
        - (3, 1, 1) - Four robots in a square
        - (4, 1, 1) - Two robots in a narrow corridor
    """
    scenario_index_list = [
        # (1, 1, 3), 
        # (1, 2, 2), 
        (1, 3, 1), 
        # (2, 1, 2), 
        # (2, 1, 3),
        # (3, 1, 1), 
        # (4, 1, 1),
    ]
    tracker_type_list = [
        'dwa-gpdf', 
        # 'dwa-esdf',
        # 'dwa',
        # 'dwa-pred', 
        # 'mpc',
    ]

    for tracker_type in tracker_type_list:
        for scenario_index in scenario_index_list:
            # tracker_type = 'dwa-gpdf' # 'mpc', 'dwa', 'dwa-pred', 'dwa-gpdf', 'rpp'
            planner_type = None # None, 'teb', 'dwa', 'dwa-pred', 'dwa-gpdf'
            # scenario_index = (2, 1, 3)
            auto_run = True
            map_only = True
            save_video_name = None#f'./Demo/{scenario_index[0]}_{scenario_index[1]}_{scenario_index[2]}_{tracker_type}_{planner_type}.avi'
            evaluation = False
            repeat = 1
            time_step = 30.0 # seconds. 200 for long-term, 50 for short-term

            project_dir = pathlib.Path(__file__).resolve().parents[1]

            ### Check planner type
            if planner_type is None:
                cfg_planner_path = 'none'
            elif planner_type == 'teb':
                cfg_planner_path = os.path.join(project_dir, 'config', 'teb.yaml')
            elif planner_type in ['dwa', 'dwa-gpdf', 'dwa-esdf', 'dwa-pred']:
                cfg_planner_path = os.path.join(project_dir, 'config', 'dwa.yaml')
            else:
                raise ValueError(f'Invalid planner type: {planner_type}')

            ### Check tracker type
            if tracker_type == 'mpc':
                cfg_tracker_path = os.path.join(project_dir, 'config', 'mpc_fast.yaml')
            elif tracker_type in ['dwa', 'dwa-gpdf', 'dwa-esdf', 'dwa-pred']:
                cfg_tracker_path = os.path.join(project_dir, 'config', 'dwa.yaml')
            elif tracker_type == 'rpp':
                cfg_tracker_path = os.path.join(project_dir, 'config', 'rpp.yaml')
            else:
                raise ValueError(f'Invalid tracker type: {tracker_type}')

            ### Other paths
            cfg_robot_path = os.path.join(project_dir, 'config', 'spec_robot.yaml')
            cfg_human_path = os.path.join(project_dir, 'config', 'spec_human.yaml')
            cfg_tf_path = os.path.join(project_dir, 'config', 'global_setting_warehouse.yaml')

            mb = MainBase(scenario_index=scenario_index, evaluation=evaluation, map_only=map_only, save_video_name=save_video_name, verbose=False)
            mb.load(planner_type, cfg_planner_path, tracker_type, cfg_tracker_path, cfg_robot_path, cfg_human_path)
            mb.prepare()

            # fig_debug, axes_debug = plt.subplots(1, 2) 

            mb.run_once(repeat=repeat, time_step_out=time_step, extra_debug_panel=None, auto_run=auto_run)
            if evaluation:
                mb.report(save_dir='./')

            print(f'Finish with tracker: {tracker_type}, planner: {planner_type}, scenario: {scenario_index}.')


    




