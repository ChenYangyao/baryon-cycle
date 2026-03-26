from __future__ import annotations
import typing
from typing import Self
from sklearn.neighbors import KDTree
import numpy as np
from pyhipp.astro.coords.cvt import AstroSpherical


class FieldInterpolator2DCartesian:
    def __init__(self, xs: np.ndarray, ys: np.ndarray, filled: np.ndarray):
        self.xs = xs
        self.ys = ys
        self.filled = filled

        self.__set_up()

    def eval(self, field: np.ndarray) -> np.ndarray:
        i0s_empty, i1s_empty = self.inds_empty
        i0s_near, i1s_near = self.inds_near

        out = field.copy()
        out[i0s_empty, i1s_empty] = field[i0s_near, i1s_near]
        return out

    def __set_up(self):
        i0s_filled, i1s_filled = self.filled.nonzero()
        i0s_empty, i1s_empty = (~self.filled).nonzero()

        xs_filled = np.column_stack((self.xs[i0s_filled], self.ys[i1s_filled]))
        xs_empty = np.column_stack((self.xs[i0s_empty], self.ys[i1s_empty]))

        ds, nears = KDTree(xs_filled, leaf_size=16).query(xs_empty, k=1)
        ds, nears = ds[:, 0], nears[:, 0]
        i0s_near, i1s_near = i0s_filled[nears], i1s_filled[nears]

        self.inds_filled = i0s_filled, i1s_filled
        self.inds_empty = i0s_empty, i1s_empty
        self.inds_near = i0s_near, i1s_near


class FieldInterpolatorND:
    def __init__(self, Xs: np.ndarray, filled: np.ndarray):
        '''
        @Xs: (N, N_d), where N_d is the number of dimensions, and N is the 
        number of points.
        
        @filled: (N,).
        '''
        self.Xs = Xs
        self.filled = filled
        self.__set_up()

    @staticmethod
    def from_astro_spherical(
            rs: np.ndarray, thetas: np.ndarray, phis: np.ndarray,
            filled: np.ndarray) -> FieldInterpolatorND:
        Xs = np.column_stack(AstroSpherical.to_cartesian(rs, thetas, phis))
        return FieldInterpolatorND(Xs, filled)

    def eval(self, field: np.ndarray) -> np.ndarray:
        inds_empty = self.inds_empty
        inds_near = self.inds_near

        out = field.copy()
        out[inds_empty] = field[inds_near]
        return out

    def __set_up(self):
        Xs, filled = self.Xs, self.filled
        empty = ~filled

        inds_filled = filled.nonzero()[0]
        inds_empty = empty.nonzero()[0]

        Xs_filled = Xs[inds_filled]
        Xs_empty = Xs[inds_empty]

        ds, nears = KDTree(Xs_filled, leaf_size=16).query(Xs_empty, k=1)
        ds, nears = ds[:, 0], nears[:, 0]
        inds_near = inds_filled[nears]

        self.inds_filled = inds_filled
        self.inds_empty = inds_empty
        self.inds_near = inds_near
