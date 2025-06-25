from typing import Union

import numpy as np
from matplotlib.axes import Axes  # type: ignore


class EuclideanSDF:
    """Euclidean Signed Distance Field with gradient information.
    
    Computes signed distance and gradient for 2D point clouds.
    Positive distance indicates outside obstacle, negative inside.
    """
    
    def __init__(self):
        self.obstacle_set:dict[Union[int, str], np.ndarray] = {} # type: ignore

    def add_obstacle(self, index:Union[int, str], obstacle_points: np.ndarray):
        self.obstacle_set[index] = obstacle_points

    def add_obstacle_after_interp(self, index:Union[int, str], obstacle_points: np.ndarray, interp_res: float = 0.1):
        """Interpolate between points to create a denser obstacle representation."""
        new_obstacle = []
        for i in range(obstacle_points.shape[0] - 1):
            p1, p2 = obstacle_points[i, :], obstacle_points[i + 1, :]
            segment = np.linspace(p1, p2, int(np.linalg.norm(p2 - p1) / interp_res) + 1)
            new_obstacle.extend(segment[:-1])
        # Connect last point to first
        p1, p2 = obstacle_points[-1, :], obstacle_points[0, :]
        segment = np.linspace(p1, p2, int(np.linalg.norm(p2 - p1) / interp_res) + 1)
        new_obstacle.extend(segment[:-1])
        self.add_obstacle(index, np.array(new_obstacle))
        return np.array(new_obstacle)
    
    def add_obstacles(self, indices: list[Union[int, str]], obstacle_points_list: list[np.ndarray]):
        """Add multiple obstacles at once."""
        if len(indices) != len(obstacle_points_list):
            raise ValueError("Indices and obstacle points list must have the same length.")
        for i, points in zip(indices, obstacle_points_list):
            self.add_obstacle(i, points)

    def add_obstacles_after_interp(self, indices: list[Union[int, str]], obstacle_points_list: list[np.ndarray], interp_res: float = 0.1):
        """Add multiple obstacles with interpolation."""
        if len(indices) != len(obstacle_points_list):
            raise ValueError("Indices and obstacle points list must have the same length.")
        new_obstacles = []
        for i, points in zip(indices, obstacle_points_list):
            new_obstacle = self.add_obstacle_after_interp(i, points, interp_res)
            new_obstacles.append(new_obstacle)
        return np.vstack(new_obstacles)


    def compute_distance(self, query_points: np.ndarray) -> np.ndarray:
        """Compute signed distances to obstacle surface.
        
        Args:
            query_points: (M,2) array of query points
            
        Returns:
            (M,) array of signed distances
        """
        if len(query_points.shape) == 1:
            query_points = query_points[np.newaxis, :]
            
        # Compute distances to all obstacle points
        obstacle_points = np.vstack(list(self.obstacle_set.values()))
        diffs = query_points[:, np.newaxis, :] - obstacle_points[np.newaxis, :, :]
        distances = np.linalg.norm(diffs, axis=2)
        
        # Find minimum distance for each query point
        min_distances = np.min(distances, axis=1)
        
        # TODO: Implement proper sign determination (inside/outside)
        # For now assuming all points are outside obstacles
        return min_distances
        
    def compute_gradient(self, query_points: np.ndarray) -> np.ndarray:
        """Compute gradient vectors at query points.
        
        Args:
            query_points: (M,2) array of query points
            
        Returns:
            (M,2) array of gradient vectors
        """
        if len(query_points.shape) == 1:
            query_points = query_points[np.newaxis, :]
            
        # Compute vectors to all obstacle points
        obstacle_points = np.vstack(list(self.obstacle_set.values()))
        diffs = query_points[:, np.newaxis, :] - obstacle_points[np.newaxis, :, :]
        distances = np.linalg.norm(diffs, axis=2)
        
        # Find nearest obstacle point for each query point
        nearest_indices = np.argmin(distances, axis=1)
        nearest_points = obstacle_points[nearest_indices]
        
        # Gradient points from query to nearest obstacle point
        gradients = nearest_points - query_points
        
        # Normalize gradient vectors
        norms = np.linalg.norm(gradients, axis=1, keepdims=True)
        gradients = np.where(norms > 0, gradients / norms, np.zeros_like(gradients))
        
        return -gradients
        
    def compute_distance_and_gradient(self, query_points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute both distance and gradient efficiently.
        
        Args:
            query_points: (M,2) array of query points
            
        Returns:
            Tuple of (distances, gradients) where:
            - distances: (M,) array of signed distances
            - gradients: (M,2) array of gradient vectors
        """
        distances = self.compute_distance(query_points)
        gradients = self.compute_gradient(query_points)
        return distances, gradients
    
    def plot_env(self, ax: Axes, x_range: tuple, y_range: tuple, map_resolution=(100, 100), color='k', plot_grad_dir=False, show_grad=False):
        _x = np.linspace(x_range[0], x_range[1], map_resolution[1])
        _y = np.linspace(y_range[0], y_range[1], map_resolution[0])
        ctr_level = 20 # default 20
        
        X, Y = np.meshgrid(_x, _y)
        dis_mat = np.zeros(X.shape)
        all_xy_coords = np.column_stack((X.ravel(), Y.ravel()))
        dis_mat, normal = self.compute_distance_and_gradient(all_xy_coords)
        quiver = None
        if plot_grad_dir:
            quiver = ax.quiver(X, Y, normal[:, 0], normal[:, 1], color='gray', scale=30, alpha=.3)
        dis_mat = dis_mat.reshape(map_resolution) - 0.0
        if show_grad:
            ctr = ax.contour(X, Y, dis_mat, levels=ctr_level, linewidths=1.5, alpha=.3)
            ctrf = ax.contourf(X, Y, dis_mat, levels=ctr_level, extend='min', alpha=.3)
            ax.clabel(ctr, inline=True)
        else:
            ctr = ax.contour(X, Y, dis_mat, [0], colors=color, linewidths=1.5)
            ctrf = ax.contourf(X, Y, dis_mat, [0, 0.1], colors=['orange','white'], extend='min', alpha=.3)
        return ctr, ctrf, quiver, dis_mat