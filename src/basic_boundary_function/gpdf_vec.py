import sys
import math
from typing import Optional, cast

import jax
import numpy as np
from PIL import Image # type: ignore

from .gpdf_w_hes import train_gpdf
from .gpdf_w_hes import infer_gpdf_dis, infer_gpdf_hes, infer_gpdf, infer_gpdf_grad

np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(suppress=True)


class GassianProcessDistanceField:
    """Gaussian Process Distance Field"""
    def __init__(self, pc_coords:Optional[np.ndarray]=None) -> None:
        self._pc_coords = pc_coords
        if pc_coords is not None:
            self.update_gpdf(pc_coords)
    
    @property
    def pc_coords(self):
        return self._pc_coords
    
    def update_gpdf(self, new_pc_coords: np.ndarray):
        """The shape of new_pc_coords is (n, 2)"""
        self._pc_coords = new_pc_coords
        self.gpdf_model = train_gpdf(self.pc_coords)
        self.gpdf_model = cast(jax.Array, self.gpdf_model)

    def dis_func(self, states):
        """ Query the distances of the states to the boundary.

        Args
            states: The states to query, (n, d).
        """
        return infer_gpdf_dis(self.gpdf_model, self.pc_coords, query=states).flatten()+1

    def normal_func(self, states):
        # states n*d
        normal = infer_gpdf_grad(self.gpdf_model, self.pc_coords, states)
        return normal

    def dis_normal_func(self, states):
        # states n*d
        dis, normal = infer_gpdf(self.gpdf_model, self.pc_coords, states)
        normal = normal.squeeze()
        return dis.flatten()+1, normal
    
    def dis_normal_hes_func(self, states):
        """ Query the distances, normals, and hessians of the states to the boundary.

        Args
            states: The states to query, (n, d).
        """
        dis, normal, hes = infer_gpdf_hes(self.gpdf_model, self.pc_coords, states)
        normal = normal.squeeze()
        return dis.flatten()+1, normal, hes
    
    @staticmethod
    def load_image_as_target(image_path):
        with Image.open(image_path) as img:
            img_gray = img.convert("L")
            target = np.array(img_gray, dtype=float)
            target = np.flipud(target)
            return target