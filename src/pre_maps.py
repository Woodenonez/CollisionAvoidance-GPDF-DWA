import math

import numpy as np


PathNode = tuple[float, float]
MapDescription = tuple[list[np.ndarray], list[tuple], list[np.ndarray], list[list[PathNode]], list[PathNode], list[list[PathNode]]]


def generate_map(scene:int=1, sub_scene:int=1, sub_scene_option:int=1) -> MapDescription:
    """
    MapDescription
        robot_starts: list of np.ndarray
        robot_goals: list of tuple
        human_starts: list of np.ndarray (empty if no human)
        human_paths: list of list of PathNode (empty if no human)
        boundary: list of PathNode
        obstacles_list: list of list of PathNode
    """
    exist_scene = []
    if scene == 1:
        map_des = generate_map_scene_1(sub_scene, sub_scene_option)
        exist_scene.append(1)
    elif scene == 2:
        map_des = generate_map_scene_2(sub_scene, sub_scene_option)
        exist_scene.append(2)
    elif scene == 3:
        map_des = generate_map_scene_3(sub_scene, sub_scene_option)
        exist_scene.append(3)
    elif scene == 4:
        map_des = generate_map_scene_4(sub_scene, sub_scene_option)
        exist_scene.append(4)
    else:
        raise ValueError(f"Scene {scene} not recognized (should be {exist_scene}).")
    
    return map_des
    

### Evaluation maps ###
"""
Scene 1: Crosswalk
    - a. Single rectangular static obstacle (small, medium, large)
    - b. Two rectangular static obstacles (small/large stagger, close/far aligned)
    - c. Single non-convex static obstacle (deep/shallow u-/v-shape)
    - d. Single dynamic obstacle (crash, cross)
    --------------------
                |   |
    R =>    S   | D |
    --------------------

Scene 2: Turning
    - a. Single rectangular obstacle (right, sharp, u-shape)
    |-------|
    |   ->  \
    | R |\   \
    
Scene 3: Multi-robot
    - a. 4 robots (empty, middle obstacle)
"""

test_scene_1_dict = {1: [1, 2, 3], 2: [1, 2, 3, 4], 3: [1, 2, 3, 4], 4: [1, 2]}
test_scene_2_dict = {1: [1, 2, 3]}


def generate_map_scene_1(sub_index: int, scene_option: int):
    """
    Subscene index (`sub_index`) with scene option (`scene_option`): 
    - 1: Single rectangular static obstacle 
        - (1-small, 2-medium, 3-large)
    - 2: Two rectangular static obstacles 
        - (1-small stagger, 2-large stagger, 3-close aligned, 4-far aligned)
    - 3: Single non-convex static obstacle
        - (1-big u-shape, 2-small u-shape, 3-big v-shape, 4-small v-shape)
    - 4: Single dynamic obstacle
        - (1-crash, 2-cross)

    Return
        robot_starts: list of np.ndarray
        robot_goals: list of tuple
        human_starts: list of np.ndarray (empty if no human)
        human_paths: list of list of PathNode (empty if no human)
        boundary: list of PathNode
        obstacles_list: list of list of PathNode
    """
    robot_starts = [np.array([0.6, 3.5, 0.0,])]
    robot_goals = [
        [(15.4, 3.5)],
    ]
    boundary = [(0.0, 0.0), (16.0, 0.0), (16.0, 10.0), (0.0, 10.0)]
    obstacles_list = [
        [(0.0, 1.5), (0.0, 1.6), (9.0, 1.6), (9.0, 1.5)],
        [(0.0, 8.4), (0.0, 8.5), (9.0, 8.5), (9.0, 8.4)],
        [(11.0, 1.5), (11.0, 1.6), (16.0, 1.6), (16.0, 1.5)],
        [(11.0, 8.4), (11.0, 8.5), (16.0, 8.5), (16.0, 8.4)],
    ]

    unexpected_obstacles = []
    human_starts: list[np.ndarray] = []  
    human_paths: list[list[PathNode]] = []

    if sub_index == 1:
        if scene_option == 1:
            unexpected_obstacle = [(7.5, 3.0), (7.5, 4.0), (8.5, 4.0), (8.5, 3.0)] # small
        elif scene_option == 2:
            unexpected_obstacle = [(7.2, 2.8), (7.2, 4.2), (8.8, 4.2), (8.8, 2.8)] # medium
        elif scene_option == 3:
            unexpected_obstacle = [(7.0, 2.5), (7.0, 4.5), (9.0, 4.5), (9.0, 2.5)] # large
        else:
            raise ValueError(f"Invalid scene {sub_index} option, should be 1~3.")
        unexpected_obstacles.append(unexpected_obstacle)
        # human_starts = [np.array([15.4, 3.5, 0.0])]  
        # human_paths = [[(15.4, 3.5), (5.0, 3.5), ]]

    elif sub_index == 2:
        if scene_option == 1:
            unexpected_obstacle_1 = [(5.0 ,1.5), (5.0, 4.0), (6.0, 4.0), (6.0, 1.5)]
            unexpected_obstacle_2 = [(8.5, 3.5), (8.5, 8.0), (9.5, 8.0), (9.5, 3.5)]
        elif scene_option == 2:
            unexpected_obstacle_1 = [(5.0, 1.5), (5.0, 5.0), (6.0, 5.0), (6.0, 1.5)]
            unexpected_obstacle_2 = [(8.5, 3.5), (8.5, 8.0), (9.5, 8.0), (9.5, 3.5)]
        elif scene_option == 3:
            unexpected_obstacle_1 = [(4.2, 2.8), (4.2, 4.2), (5.8, 4.2), (5.8, 2.8)]
            unexpected_obstacle_2 = [(6.2, 2.8), (6.2, 4.2), (7.8, 4.2), (7.8, 2.8)]
        elif scene_option == 4:
            unexpected_obstacle_1 = [(4.2, 2.8), (4.2, 4.2), (5.8, 4.2), (5.8, 2.8)]
            unexpected_obstacle_2 = [(8.2, 2.8), (8.2, 4.2), (9.8, 4.2), (9.8, 2.8)]
        else:
            raise ValueError(f"Invalid scene {sub_index} option, should be 1~4.")
        unexpected_obstacles.append(unexpected_obstacle_1)
        unexpected_obstacles.append(unexpected_obstacle_2)

    elif sub_index == 3:
        unexpected_obstacle_3 = None
        if scene_option == 1:
            unexpected_obstacle_1 = [(6.0, 4.5), (6.0, 5.0), (8.5, 5.0), (8.5, 4.5)]
            unexpected_obstacle_2 = [(8.5, 5.0), (8.5, 2.0), (8.0, 2.0), (8.0, 5.0)]
            unexpected_obstacle_3 = [(8.5, 2.0), (6.0, 2.0), (6.0, 2.5), (8.5, 2.5)]
        elif scene_option == 2:
            unexpected_obstacle_1 = [(6.0, 4.0), (6.0, 4.5), (7.5, 4.5), (7.5, 4.0)]
            unexpected_obstacle_2 = [(7.5, 4.5), (7.5, 2.0), (7.0, 2.0), (7.0, 4.5)]
            unexpected_obstacle_3 = [(7.5, 2.0), (6.0, 2.0), (6.0, 2.5), (7.5, 2.5)]
        elif scene_option == 3:
            unexpected_obstacle_1 = [(6.0, 5.0), (9.5, 5.0), (9.5, 3.5), (9.0, 3.5)]
            unexpected_obstacle_2 = [(9.5, 3.5), (9.5, 2.0), (6.0, 2.0), (9.0 ,3.5)]
        elif scene_option == 4:
            unexpected_obstacle_1 = [(6.5, 4.5), (8.5, 4.5), (8.5, 3.5), (8.0, 3.5)]
            unexpected_obstacle_2 = [(8.5, 3.5), (8.5, 2.5), (6.5, 2.5), (8.0, 3.5)]
        else:
            raise ValueError(f"Invalid scene {sub_index} option, should be 1~4.")
        unexpected_obstacles.append(unexpected_obstacle_1)
        unexpected_obstacles.append(unexpected_obstacle_2)
        if unexpected_obstacle_3 is not None:
            unexpected_obstacles.append(unexpected_obstacle_3)

    elif sub_index == 4:
        raise NotImplementedError # XXX
        # if scene_option == 1:
        #     unexpected_obstacle = Obstacle.create_mpc_dynamic(p1=(15.4, 3.5), p2=(0.6, 3.5), freq=0.15, rx=0.8, ry=0.8, angle=0.0, corners=20)
        #     unexpected_obstacles.append(unexpected_obstacle)
        # elif scene_option == 2:
        #     unexpected_obstacle = Obstacle.create_mpc_dynamic(p1=(10.0, 1.0), p2=(10.0, 9.0), freq=0.2, rx=0.8, ry=0.8, angle=0.0, corners=20)
        #     unexpected_obstacles.append(unexpected_obstacle)
        # else:
        #     raise ValueError(f"Invalid scene {sub_index} option, should be 1~2.")

        human_starts = [np.array([110, 20, 0.0])]  
        human_paths = [[(0.0, 0.0), (0.0, 10.0), ]]
    
    else:
        raise ValueError(f"Invalid scene index, should be 1~4.")

    obstacles_list.extend(unexpected_obstacles)

    return robot_starts, robot_goals, human_starts, human_paths, boundary, obstacles_list

def generate_map_scene_2(sub_index: int, scene_option: int):
    """
    Subscene index (`sub_index`) with scene option (`scene_option`): 
    - 1: Single rectangular obstacle
        - (1-right, 2-sharp, 3-u-shape)
    - 2: Single dynamic obstacle
        - (1-right, 2-sharp, 3-u-shape)

    Return
        robot_starts: list of np.ndarray
        robot_goals: list of tuple
        human_starts: list of np.ndarray (empty if no human)
        human_paths: list of list of PathNode (empty if no human)
        boundary: list of PathNode
        obstacles_list: list of list of PathNode
    """
    robot_starts = [np.array([3.0, 0.6, math.pi/2])]
    robot_goals_1 = [
        [(3.0, 14.0), (15.5, 14.0)],
    ]
    robot_goals_2 = [
        [(3.0, 13.5), (5.5, 13.5), (11.0, 0.6)],
    ]
    robot_goals_3 = [
        [(3.0, 13.5), (5.5, 13.5), (5.5, 0.6)],
    ]
    boundary = [(0.0, 0.0), (16.0, 0.0), (16.0, 18.0), (0.0, 18.0)]
    obstacles_list = [[(0.0, 0.0), (0.0, 16.0), (1.0, 16.0), (1.0, 0.0)], 
    ]
                      
    more_obstacles = []
    unexpected_obstacles = []
    human_starts: list[np.ndarray] = []  
    human_paths: list[list[PathNode]] = []

    if sub_index == 1:
        if scene_option == 1:
            robot_goals = robot_goals_1
            more_obstacle_1 = [(4.0, 0.0), (16.0, 0.0), (16.0, 13.0), (4.0, 13.0)] # right
            more_obstacle_2 = None
            unexpected_obstacle = [(3.0, 14.0), (4.0, 14.0), (4.0, 13.0), (3.0, 13.0)]
        elif scene_option == 2:
            robot_goals = robot_goals_2
            more_obstacle_1 = [(4.0, 0.0), (4.0, 13.0), (4.5, 13.0), (10.0, 0.0)] # sharp
            more_obstacle_2 = [(15.0, 0.0), (16.0, 0.0), (16.0, 16.0), (8.0, 16.0)] # sharp
            unexpected_obstacle = [(4.0, 13.5), (4.0, 14.0), (4.5, 14.0), (4.5, 13.5)]
        elif scene_option == 3:
            robot_goals = robot_goals_3
            more_obstacle_1 = [(4.0, 0.0), (4.0, 13.0), (4.5, 13.0), (4.5, 0.0)] # u-turn
            more_obstacle_2 = [(7.5, 0.0), (16.0, 0.0), (16.0, 16.0), (7.5, 16.0)] # u-turn
            unexpected_obstacle = [(4.0, 13.5), (4.0, 14.0), (4.5, 14.0), (4.5, 13.5)]
        else:
            raise ValueError(f"Invalid scene {sub_index} option, should be 1~3.")
        more_obstacles.append(more_obstacle_1)
        if more_obstacle_2 is not None:
            more_obstacles.append(more_obstacle_2)
        unexpected_obstacles.append(unexpected_obstacle)

    elif sub_index == 2:
        raise NotImplementedError
        if scene_option == 1:
            robot_goals = robot_goals_2
            # more_obstacle_1 = Obstacle.create_mpc_static([(4.0, 0.0), (4.0, 13.0), (4.5, 13.0), (10.0, 0.0)]) # sharp
            # more_obstacle_2 = Obstacle.create_mpc_static([(15.0, 0.0), (16.0, 0.0), (16.0, 16.0), (8.0, 16.0)]) # sharp
            # unexpected_obstacle = Obstacle.create_mpc_static([(4.0, 13.5), (4.0, 14.0), (4.5, 14.0), (4.5, 13.5)])
        else:
            raise ValueError(f"Invalid scene {sub_index} option, should be 1.")
        # more_obstacles.append(more_obstacle_1)
        # more_obstacles.append(more_obstacle_2)
        # unexpected_obstacles.append(unexpected_obstacle)
    
    else:
        raise ValueError(f"Invalid scene index, should be 1~4.")

    obstacles_list.extend(unexpected_obstacles)
    obstacles_list.extend(more_obstacles)

    return robot_starts, robot_goals, human_starts, human_paths, boundary, obstacles_list

def generate_map_scene_3(sub_index: int, scene_option: int):
    """
    Subscene index (`sub_index`) with scene option (`scene_option`):
    - 1: 4 robots
        - (1-empty, 2-middle obstacle)

    Return
        robot_starts: list of np.ndarray
        robot_goals: list of tuple
        human_starts: list of np.ndarray (empty if no human)
        human_paths: list of list of PathNode (empty if no human)
        boundary: list of PathNode
        obstacles_list: list of list of PathNode
    """
    robot_starts = [
        np.array([0.0, 0.0, math.radians(45)]),
        np.array([10.0, 10.0, math.radians(-135)]),
        np.array([0.0, 10.0, math.radians(-45)]),
        np.array([10.0, 0.0, math.radians(135)]),
    ]
    robot_goals = [
        [(10.0, 10.0)],
        [(0.0, 0.0)],
        [(10.0, 0.0)],
        [(0.0, 10.0)],
    ]
    boundary = [(-5.0, -5.0), (15.0, -5.0), (15.0, 15.0), (-5.0, 15.0)]
    obstacles_list = []

    human_starts: list[np.ndarray] = []  
    human_paths: list[list[PathNode]] = []

    if sub_index == 1:
        if scene_option == 1:
            pass
        elif scene_option == 2:
            obstacles_list.append(
                [(5.0, 3.5), (6.5, 5.0), (5.0, 6.5), (3.5, 5.0)]
            )

    return robot_starts, robot_goals, human_starts, human_paths, boundary, obstacles_list

def generate_map_scene_4(sub_index: int, scene_option: int):
    """Parallel tunnel + 2 robots
    """

    robot_starts = [
        np.array([8.5, 5.0, math.radians(180)]),
        np.array([-1.0, 5.0, math.radians(0)]),
    ]
    robot_goals = [
        [(-1.0, 5.0)],
        [(9.0, 5.0)],
    ]

    boundary = [(-3.0, -1.0), (-3.0, 11.0), (11.0, 11.0), (11.0, -1.0)]
    obstacles_list = [[(3.0, 5.85), (3.0, 6.0), (7.0, 6.0), (7.0, 5.85)],
                      [(3.0, 8.85), (3.0, 9.0), (7.0, 9.0), (7.0, 8.85)],
                      [(-2.0, 0.0), (-2.0, 4.0), (9.0, 4.0), (9.0, 0.0)]]
    
    human_starts: list[np.ndarray] = []  
    human_paths: list[list[PathNode]] = []

    return robot_starts, robot_goals, human_starts, human_paths, boundary, obstacles_list
