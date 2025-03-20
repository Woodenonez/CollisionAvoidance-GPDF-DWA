import numpy as np
from matplotlib.axes import Axes # type: ignore

from .gpdf_vec import GassianProcessDistanceField as GPDF # type: ignore


class GPDFEnv:
    def __init__(self, margin=0.0, rho=10.0) -> None:
        self.margin = margin
        self.rho = rho

        self.gpdf_set = {} # type: ignore

    @property
    def num_gpdf(self):
        return len(self.gpdf_set)
    
    def add_gpdf(self, index, gpdf=None, pc_coords=None):
        if gpdf is not None:
            self.gpdf_set[index] = gpdf
        elif pc_coords is not None:
            new_gpdf = GPDF()
            new_gpdf.update_gpdf(pc_coords)
            self.gpdf_set[index] = new_gpdf
        else:
            raise ValueError
        
    def add_gpdf_after_interp(self, index, pc_coords, interp_res=0.1):
        new_pc_coords = []
        for i in range(pc_coords.shape[0]-1):
            p1, p2 = pc_coords[i, :], pc_coords[i+1, :]
            segment = np.linspace(p1, p2, int(np.linalg.norm(p2-p1)/interp_res)+1)
            new_pc_coords.extend(segment[:-1])
        p1, p2 = pc_coords[-1, :], pc_coords[0, :]
        segment = np.linspace(p1, p2, int(np.linalg.norm(p2-p1)/interp_res)+1)
        new_pc_coords.extend(segment[:-1])
        self.add_gpdf(index, pc_coords=np.array(new_pc_coords))
        return np.array(new_pc_coords)

        
    def add_gpdfs(self, indices, gpdfs=None, pc_coords_list=None):
        if gpdfs is not None:
            for i, gpdf in zip(indices, gpdfs):
                self.gpdf_set[i] = gpdf
        elif pc_coords_list is not None:
            for i, pc in zip(indices, pc_coords_list):
                new_gpdf = GPDF()
                new_gpdf.update_gpdf(pc)
                self.gpdf_set[i] = new_gpdf
        else:
            raise ValueError

    def add_gpdfs_after_interp(self, indices, pc_coords_list, interp_res=0.1):
        for i, pc_coords in zip(indices, pc_coords_list):
            self.add_gpdf_after_interp(i, pc_coords, interp_res) 
        
    def remove_gpdf(self, index):
        del self.gpdf_set[index]

    def h_grad_set(self, x):
        for k in range(self.num_gpdf):
            dis, grad  = self.gpdf_set[k].dis_normal_func(x)
            if k==0:
                dis_set = np.asarray(dis).reshape(1, -1)
                grad_set = np.asarray(grad).reshape(1, 2)
            else:
                dis_set = np.concatenate((dis_set, np.asarray(dis).reshape(1, -1)),axis=0)
                grad_set = np.concatenate((grad_set, np.asarray(grad).reshape(1, 2)), axis=0)
        return dis_set, grad_set

    def h_grad_vector(self, x, obstacle_idx=-1, exclude_index=None):
        if isinstance(obstacle_idx, int) and obstacle_idx < 0: # all obstacles
            dis_set = np.zeros((self.num_gpdf, len(x)))
            grad_set = np.zeros((self.num_gpdf, len(x), 2))
            gpdf_keys = [x for x in list(self.gpdf_set) if x!=exclude_index]
            for i, key in enumerate(gpdf_keys):
                dis, grad = self.gpdf_set[key].dis_normal_func(x)
                if len(grad.shape)==1:
                    grad = grad.reshape(2, -1)
                if i==0:
                    dis_set = np.asarray(dis).reshape(1, -1)
                    grad_set = np.asarray(grad).T[None, :, :]
                else:
                    dis_set = np.concatenate((dis_set, np.asarray(dis).reshape(1, -1)),axis=0)
                    grad_set = np.concatenate((grad_set, grad.T[None, :, :]), axis=0)
        else:
            dis, grad  = self.gpdf_set[obstacle_idx].dis_normal_func(x)
            dis_set = np.asarray(dis).reshape(1,-1)
            grad_set = grad.T[None, :, :]

        grad_num = np.sum(np.exp(-self.rho*dis_set[:, :, None])*grad_set, axis=0)
        grad_den = np.sum(np.exp(-self.rho*dis_set), axis=0)
        with np.errstate(divide='ignore', invalid='ignore'):
            dis_uni = -1/self.rho*np.log(grad_den)-1
            grad_uni = grad_num/grad_den.reshape(-1,1)
        return dis_uni, grad_uni

    def plot_env(self, ax: Axes, x_range, y_range, map_resolution=(100, 100), color='k', plot_grad_dir=False, obstacle_idx=-1, show_grad=False, exclude_index=None):
        _x = np.linspace(x_range[0], x_range[1], map_resolution[0])
        _y = np.linspace(y_range[0], y_range[1], map_resolution[1])
        X, Y = np.meshgrid(_x, _y)
        dis_mat = np.zeros(X.shape)
        all_xy_coords = np.column_stack((X.ravel(), Y.ravel()))
        dis_mat, normal = self.h_grad_vector(all_xy_coords, obstacle_idx=obstacle_idx, exclude_index=exclude_index)
        quiver = None
        if plot_grad_dir:
            quiver = ax.quiver(X, Y, normal[:, 0], normal[:, 1], color='gray', scale=30, alpha=.3)
        dis_mat = dis_mat.reshape(map_resolution) - 0.0
        if show_grad:
            ctr = ax.contour(X, Y, dis_mat, levels=20, linewidths=1.5, alpha=.3)
            ctrf = ax.contourf(X, Y, dis_mat, levels=20, extend='min', alpha=.3)
            ax.clabel(ctr, inline=True)
        else:
            ctr = ax.contour(X, Y, dis_mat, [0], colors=color, linewidths=1.5)
            ctrf = ax.contourf(X, Y, dis_mat, [0, 0.1], colors=['orange','white'], extend='min', alpha=.3)
        return ctr, ctrf, quiver
    