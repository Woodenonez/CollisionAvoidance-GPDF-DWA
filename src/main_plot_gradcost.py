from typing import Optional

import numpy as np
import matplotlib.pyplot as plt # type: ignore
from matplotlib.axes import Axes # type: ignore
import matplotlib.cm as cm # type: ignore
import matplotlib.colors as mcolors # type: ignore

from shapely.geometry import Point, Polygon # type: ignore

from basic_boundary_function.env import GPDFEnv


colors = [(1, 0, 0, alpha) for alpha in np.linspace(0, 1, 256)]  # RGBA (Red, Green, Blue, Alpha)
custom_cmap = mcolors.LinearSegmentedColormap.from_list("TransparentRed", colors, N=256)


def calc_cost_gpdf(traj: np.ndarray, dist_set: np.ndarray, grad_set: np.ndarray, min_safe_dist:float=0.0):
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
    sa_cost = (sa_cost - sa_min) / (sa_max - sa_min)
    # sa_cost[dist_set > 1.0] = 0.0
    # sa_cost = np.sum(sa_cost) if len(sa_cost[sa_cost!=0]) > 0 else 0.0
    return sa_cost


u_shape_obstacle = [
    (0.0, -1.5), (2.5, -1.5), (2.5, 1.5),  (0.0, 1.5), 
    (0.0, 1.0),  (2.0, 1.0),  (2.0, -1.0), (0.0, -1.0)
]
obstacle_shapely = Polygon(u_shape_obstacle)
boundary = [
    (-1.9, -2.0), (3.0, -2.0), (3.0, 2.0),  (-1.9, 2.0), 
]

gpdf_env = GPDFEnv()
new_points = gpdf_env.add_gpdf_after_interp(index=f'u_shape', pc_coords=np.array(u_shape_obstacle), interp_res=0.1)

_x = np.linspace(min([x[0] for x in boundary]), max([x[0] for x in boundary]), 16)
_y = np.linspace(min([x[1] for x in boundary]), max([x[1] for x in boundary]), 15)
X, Y = np.meshgrid(_x, _y)
all_xy_coords = np.column_stack((X.ravel(), Y.ravel()))
all_states = np.repeat(all_xy_coords, 39, axis=0)
theta = np.linspace(-np.pi, np.pi, 39)
all_states = np.column_stack((all_states, np.tile(theta, len(all_xy_coords))))

dist_set_all, grad_set_all = gpdf_env.h_grad_vector(np.asarray(all_states)[:, :2])
sa_cost = calc_cost_gpdf(all_states, dist_set_all, grad_set_all)
sa_cost_normalized = (sa_cost - np.min(sa_cost)) / (np.max(sa_cost) - np.min(sa_cost))

### Optimize the visualization
sa_cost_toplot = sa_cost_normalized[sa_cost_normalized>0.01]
all_states_toplot = all_states[sa_cost_normalized>0.01] 
all_states_toplot = np.vstack((all_states_toplot, np.array([[0.0, 0.0, 0.0]])))
sa_cost_toplot = np.append(sa_cost_toplot, 0.0)
sa_cost_toplot = sa_cost_toplot[~np.array([obstacle_shapely.contains(Point(x, y)) for x, y, _ in all_states_toplot])]
all_states_toplot = all_states_toplot[~np.array([obstacle_shapely.contains(Point(x, y)) for x, y, _ in all_states_toplot])]

fig, ax = plt.subplots(figsize=(13, 8))

ax.quiver(
    all_states_toplot[:, 0], all_states_toplot[:, 1], 
    np.cos(all_states_toplot[:, 2]), np.sin(all_states_toplot[:, 2]), 
    sa_cost_toplot, cmap=custom_cmap, scale=20,
    # headaxislength=0, headlength=0, headwidth=0,
)

ctr, ctrf, map_quiver, _ = gpdf_env.plot_env(
    ax,
    x_range=(min([x[0] for x in boundary]), max([x[0] for x in boundary])),
    y_range=(min([x[1] for x in boundary]), max([x[1] for x in boundary])),
    map_resolution=(100, 100),
    color='k',
    plot_grad_dir=False,
    obstacle_idx=-1,
    show_grad=True,
)

ax.plot(np.array(u_shape_obstacle+[u_shape_obstacle[0]])[:, 0], np.array(u_shape_obstacle+[u_shape_obstacle[0]])[:, 1], 
        color='black', linewidth=2)
ax.plot(np.array(boundary+[boundary[0]])[:, 0], np.array(boundary+[boundary[0]])[:, 1], 'k--')
# ax.plot(np.array(new_points)[:, 0], np.array(new_points)[:, 1], 'rx')

cbar = fig.colorbar(ctrf, ax=ax)
cbar.ax.tick_params(labelsize=16)
# cbar.set_label("Distance")

plt.tight_layout()

ax.axis('equal')
plt.show()