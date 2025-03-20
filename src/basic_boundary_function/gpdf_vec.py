import sys

import numpy as np
from PIL import Image # type: ignore

from .gpdf_w_hes import train_gpdf
from .gpdf_w_hes import infer_gpdf_dis, infer_gpdf_hes, infer_gpdf, infer_gpdf_grad

np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(suppress=True)


class GassianProcessDistanceField:
    def __init__(self, pc_coords=None) -> None:
        self._pc_coords = pc_coords
        if pc_coords is not None:
            self.update_gpdf(pc_coords)
    
    @property
    def pc_coords(self):
        return self._pc_coords
    
    def update_gpdf(self, new_pc_coords):
        self._pc_coords = new_pc_coords
        self.gpdf_model = train_gpdf(self.pc_coords)

    def dis_func(self, states):
        return infer_gpdf_dis(self.gpdf_model, self.pc_coords, query=states).flatten()+1

    def normal_func(self, states):
        normal = infer_gpdf_grad(self.gpdf_model, self.pc_coords, states)
        return normal

    def dis_normal_func(self, states):
        dis, normal = infer_gpdf(self.gpdf_model, self.pc_coords, states)
        normal = normal.squeeze()
        return dis.flatten()+1, normal
    
    def dis_normal_hes_func(self, states):
        dis, normal, hes = infer_gpdf_hes(self.gpdf_model, self.pc_coords, states)
        normal = normal.squeeze()
        return dis.flatten()+1, normal, hes