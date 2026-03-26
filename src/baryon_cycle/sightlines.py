from __future__ import annotations
import typing
from typing import Self
import numpy as np
from pyhipp.astro.coords.cvt import PhysicalSpherical
from pyhipp.numerical.interpolate import bisearch_array
from pyhipp.stats import Rng

class TiltedDiskLocalFrame:
    def __init__(self, r_p: np.ndarray, axes: np.ndarray):
        '''
        @axes: rows are basis vectors.
        '''
        # check orthonormality
        axes = np.array(axes)
        axes_T = axes.T
        
        assert (np.abs(axes @ axes_T - np.eye(3)) < 1.0e-6).all()
        
        self.r_p = np.array(r_p)
        self.axes = axes
        self.axes_T = axes_T
        
    @staticmethod
    def new_random(rng: Rng, r_max=1500.0):
        x_p, y_p = rng.uniform(-r_max, r_max, size=2)
        r_p = np.array([x_p, y_p, 0.0])
        
        e_z = np.asarray(rng.uniform_sphere(cartesian=True))
        temp_e_y = np.array([0.0, 1.0, 0.0])
        if (np.abs(e_z - temp_e_y) < 1.0e-6).all():
            temp_e_y = np.array([1.0, 0.0, 0.0])
        e_x = np.cross(temp_e_y, e_z)
        e_x /= np.linalg.norm(e_x)
        e_y = np.cross(e_z, e_x)
        
        axes = np.array([e_x, e_y, e_z])
        
        return TiltedDiskLocalFrame(r_p, axes)

    def cart_cvt_from_cart(self, x: np.ndarray) -> np.ndarray:
        '''
        @x: shape (..., 3)
        '''
        return np.matmul(x - self.r_p, self.axes_T)

    def rp_z_cvt_from_cart(self, x: np.ndarray) -> np.ndarray:
        x1, y1, z1 = self.cart_cvt_from_cart(x).T
        rp1 = np.sqrt(x1**2 + y1**2)
        return rp1, z1

    def sph_cvt_from_cart(self, x: np.ndarray) -> np.ndarray:
        '''
        @x: shape (..., 3)
        '''
        x1 = self.cart_cvt_from_cart(x)
        r, theta, phi = PhysicalSpherical.from_cartesian(
            x1[..., 0], x1[..., 1], x1[..., 2])
        return r, theta, phi

    def r_theta_from_cart(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        '''
        Returns theta in [0, pi].
        '''

        dx = x - self.r_p
        r = np.linalg.norm(dx, axis=-1)
        dx_normed = dx / r[:, None]
        cos_theta = np.matmul(dx_normed, self.axes[2])
        theta = np.arccos(cos_theta)
        return r, theta

    def angle_r_p2ez_p(self):
        ez = self.axes[2]
        ez_p = np.array([ez[0], ez[1], 0.0])
        ez_p_norm = np.linalg.norm(ez_p)
        if ez_p_norm < 1.0e-6:
            ez_p = np.array([1.0, 0.0, 0.0])
        else:
            ez_p = ez_p/ez_p_norm
        
        r_p = self.r_p
        r_p_norm = np.linalg.norm(r_p)
        if r_p_norm < 1.0e-6:
            r_p = np.array([1.0, 0.0, 0.0])
        else:
            r_p = r_p/r_p_norm
        
        cos_angle = np.dot(ez_p, r_p)
        return np.arccos(cos_angle)

class SightlineIntegrator3DSlicedField:
    def __init__(self, local_frame: TiltedDiskLocalFrame,
                 field: np.ndarray,
                 rp_es: np.ndarray, z_es: np.ndarray):
        '''
        @field: shape (N_rps, N_zs).
        @rp_es, z_es: edges of the bins.
        '''

        self.local_frame = local_frame
        self.field = field
        self.rp_es = rp_es
        self.z_es = z_es

    def integrate(self, z_max=1500.0, eps=.1) -> float:
        zs = np.linspace(-z_max, z_max, int(2.0*z_max/eps)+1)
        dz = zs[1] - zs[0]

        X = np.column_stack((np.zeros_like(zs), np.zeros_like(zs), zs))
        rp1s, z1s = self.local_frame.rp_z_cvt_from_cart(X)

        n_rps, n_zs = self.field.shape
        i_rs = (np.searchsorted(self.rp_es, rp1s) - 1).clip(0, n_rps-1)
        i_zs = (np.searchsorted(self.z_es, z1s) - 1).clip(0, n_zs-1)

        return (self.field[i_rs, i_zs] * dz).sum()


class SightlineIntegrator:
    def __init__(self, local_frame: TiltedDiskLocalFrame,
                 field: np.ndarray,
                 r_es: np.ndarray, theta_es: np.ndarray,
                 phi_es: np.ndarray):
        '''
        @field: shape (N_r, N_theta, N_phi).
        '''

        self.local_frame = local_frame
        self.field = field
        self.r_es = r_es
        self.theta_es = theta_es
        self.phi_es = phi_es
        
        for i, es in enumerate((r_es, theta_es, phi_es)):
            assert len(es) - 1 == field.shape[i]
            assert np.diff(es).min() > 0.0, \
                f'bin edges must be strictly increasing'

    def integrate(self, z_max=1500.0, eps=.1) -> float:
        zs = np.linspace(-z_max, z_max, int(2.0*z_max/eps)+1)
        dz = zs[1] - zs[0]

        zeros = np.zeros_like(zs)
        X = np.column_stack((zeros, zeros, zs))
        rs, thetas, phis = self.local_frame.sph_cvt_from_cart(X)

        n_rs, n_thetas, n_phis = self.field.shape
        _i_rs = bisearch_array(self.r_es, rs)
        i_rs = _i_rs.clip(0, n_rs-1)
        i_thetas = bisearch_array(self.theta_es, thetas).clip(0, n_thetas-1)
        i_phis = bisearch_array(self.phi_es, phis).clip(0, n_phis-1)
        
        dIs = self.field[i_rs, i_thetas, i_phis] * dz
        dIs[_i_rs >= n_rs] = 0.

        return dIs.sum()
