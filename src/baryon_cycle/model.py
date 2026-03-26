from __future__ import annotations
import typing
from typing import Self
from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize
from .ecosys_info import EcosysInfo
from pyhipp.core import DataDict

def fn_P(x, x_0, delta_x, alpha):
    ap1 = alpha + 1.
    return (delta_x**ap1 - np.abs(x_0-x)**ap1)/delta_x**ap1

def fn_S(x, x_0, sigma, alpha):
    ap1 = alpha + 1.
    dx = np.abs(x - x_0) / sigma
    return np.exp(-dx**ap1)

def fn_double_S(x, x_0, sigma_lo, alpha_lo, sigma_hi, alpha_hi):
    is_lo = x < x_0
    is_hi = ~is_lo
    y = np.zeros_like(x)
    y[is_lo] = fn_S(x[is_lo], x_0, sigma_lo, alpha_lo)
    y[is_hi] = fn_S(x[is_hi], x_0, sigma_hi, alpha_hi)
    return y

def fn_double_PS(x, x_0, sigma_lo, alpha_lo, sigma_hi, alpha_hi):
    is_lo = x < x_0
    is_hi = ~is_lo
    y = np.zeros_like(x)
    y[is_lo] = fn_P(x[is_lo], x_0, sigma_lo, alpha_lo)
    y[is_hi] = fn_S(x[is_hi], x_0, sigma_hi, alpha_hi)
    return y

@dataclass(frozen=True, slots=True)
class FittedSpinProfile:
    y_pk: float 
    x_pk: float 
    a_d: float 
    a_pl: float 
    y_pl: float 
    s_pl: float
    
    def y_at(self, x: np.ndarray|float):
        return self._y_at(x, *self.as_tuple())
    
    @staticmethod
    def _y_at(x, y_pk, x_pk, a_d, a_pl, y_pl, s_pl):
        is_scalar = np.isscalar(x)
        x = np.array(x)
        is_lo = x < x_pk
        y_lo = y_pk * fn_P(x, x_pk, x_pk, a_d)
        y_hi = y_pl + (y_pk - y_pl) * fn_S(x, x_pk, s_pl, a_pl)
        
        y = y_hi
        y[is_lo] = y_lo[is_lo]
        if is_scalar:
            y = y.item()
        return y
    
    def as_tuple(self):
        return self.y_pk, self.x_pk, self.a_d, self.a_pl, self.y_pl, self.s_pl

class SpinFit:
    def __init__(self, xs: np.ndarray, ys: np.ndarray):
        '''
        Here x is normalized by R_h.
        '''
        self.xs = np.array(xs) 
        self.ys = np.array(ys)
        
        arg_max = ys.argmax()
        self.init_pf = FittedSpinProfile(
            y_pk=ys[arg_max],
            x_pk =xs[arg_max],
            a_d = 1. ,
            a_pl = 1.,
            y_pl = ys[xs>.5].mean(),
            s_pl = 0.1,
        )
        
    def find_optim(self):
        param0 = self.init_pf.as_tuple()
        out = minimize(self.f_obj, param0)
        if not out.success:
            print(out.message)
        self.opt_pf = FittedSpinProfile(*out.x)
        
    def f_obj(self, params):
        pf = FittedSpinProfile(*params)
        ys_pred = pf.y_at(self.xs)
        res = self.ys - ys_pred
        return (res*res).sum()
    
@dataclass(frozen=True, slots=True)
class FittedVinProfile:
    y_cool: float
    alpha_cool: float
    x_bar: float
    sigma_bar: float
    alpha_bar: float
    y_d: float
    x_d: float
    beta_d: float
    alpha_d: float
    
    def y_at(self, x: np.ndarray|float):
        return self._y_at(x, *self.as_tuple())
    
    @staticmethod
    def _y_at(x, y_cool, alpha_cool, 
              x_bar, sigma_bar, alpha_bar,
              y_d, x_d, beta_d, alpha_d,
              ):
        
        is_scalar = np.isscalar(x)
        x = np.array(x)
        x_d = max(x_d, 3.0e-3)
        sigma_bar = max(sigma_bar, 3.0e-3)
        
        s = fn_P(x.clip(0., x_bar), x_bar, sigma_bar, alpha_bar).clip(0., 1.)
        y_cgm = y_cool * x**(-alpha_cool) * s
        
        x2xd = x / x_d
        y_d = y_d * fn_S(x, 0., x_d, beta_d) * x2xd**alpha_d
        
        y = y_cgm + y_d
        
        if is_scalar:
            y = y.item()
        
        return y
    
    def as_tuple(self):
        return (self.y_cool,
            self.alpha_cool,
            self.x_bar,
            self.sigma_bar,
            self.alpha_bar,
            self.y_d,
            self.x_d,
            self.beta_d,
            self.alpha_d,)

class VinFit:
    def __init__(self, xs: np.ndarray, ys: np.ndarray):
        '''
        Here x, y is normalized by R_h and V_h, respectively. y does not 
        include Hubble flow.
        '''
        self.xs = np.array(xs) 
        self.ys = np.array(ys)
        
        self.init_pf = FittedVinProfile(
            y_cool = 0.3,
            alpha_cool = 0.5,
            x_bar = 0.7,
            sigma_bar = 0.5,
            alpha_bar=1.,
            y_d = 0.1,
            x_d = 0.03,
            beta_d = 0.,
            alpha_d=0.25,
        )
        
    def find_optim(self):
        param0 = self.init_pf.as_tuple()
        out = minimize(self.f_obj, param0, )
        if not out.success:
            print(out.message)
        self.opt_pf = FittedVinProfile(*out.x)
        
    def f_obj(self, params):
        pf = FittedVinProfile(*params)
        ys_pred = pf.y_at(self.xs)
        res = self.ys - ys_pred
        prior = (pf.beta_d - 0.0)**2 + (pf.alpha_d-0.25)**2
        return (res*res).sum() + prior

class FaceOnRecon:
    def __init__(self, bin_edges: tuple[np.ndarray,...],
                 spin_pf: FittedSpinProfile,
                 vin_pf: FittedVinProfile,
                 ecosys_info: EcosysInfo):
        
        x_es, y_es = bin_edges
        x_cs = 0.5 * (x_es[1:] + x_es[:-1])
        y_cs = 0.5 * (y_es[1:] + y_es[:-1])
        x_mcs, y_mcs = np.meshgrid(x_cs, y_cs)
        
        R_h, redshift, V_h, M_h, sim_info = (
            ecosys_info.R_h, ecosys_info.z, ecosys_info.V_h, 
            ecosys_info.M_h, ecosys_info.sim_info)
        
        self.x_es = x_es
        self.y_es = y_es
        self.x_cs = x_cs
        self.y_cs = y_cs
        self.x_mcs = x_mcs
        self.y_mcs = y_mcs
        
        self.R_h = R_h
        self.M_h = M_h
        self.V_h = V_h
        self.redshift = redshift
        self.sim_info = sim_info
        
        self.spin_pf = spin_pf
        self.vin_pf = vin_pf
        
    def run(self):
        sim_info = self.sim_info
        cosm = sim_info.cosmology
        R_h, V_h = self.R_h, self.V_h
        Ez = cosm.big_hubble(self.redshift) / cosm.big_hubble0
        H = cosm.hubble * 100.0 * Ez / 1.0e3     # km/s/kpc
        
        x_mcs, y_mcs = self.x_mcs, self.y_mcs
        rs = np.hypot(x_mcs, y_mcs)
        xs = rs / self.R_h
        
        cos_phis = x_mcs / rs
        sin_phis = y_mcs / rs
        e_phis = -sin_phis, cos_phis
        e_rs = cos_phis, sin_phis
        
        lambs = self.spin_pf.y_at(xs)
        js = lambs * np.sqrt(2.) * R_h * V_h
        v_phis = js / rs
        
        v_r = -self.vin_pf.y_at(xs) * V_h + rs * H
        
        v_xs = v_phis * e_phis[0] + v_r * e_rs[0]
        v_ys = v_phis * e_phis[1] + v_r * e_rs[1]
        
        self.data = DataDict({
            'v_phis': v_phis,
            'v_xs': v_xs,
            'v_ys': v_ys
        })
        

@dataclass(frozen=True, slots=True)
class FittedVoutProfile:
    y_pk: float
    x_pk: float
    s_1: float
    a_1: float
    s_2: float
    a_2: float
    y_plat: float
    
    def y_at(self, x: np.ndarray|float):
        y_pk, x_pk, s_1, a_1, s_2, a_2, y_plat = self.as_tuple()
        
        s_1 = max(s_1, 5.0e-3)
        s_2 = max(s_2, 1.0e-2)
        
        is_scalar = np.isscalar(x)
        x = np.asarray(x)
        
        is_lo = x < x_pk
        #y_lo = y_pk * fn_P(x, x_pk, s_1, a_1).clip(0.)
        y_lo = y_pk * fn_S(x, x_pk, s_1, a_1)
        y_hi = (y_pk-y_plat) * fn_S(x, x_pk, s_2, a_2) + y_plat
        
        y = y_hi
        y[is_lo] = y_lo[is_lo]
        if is_scalar:
            y = y.item()
        return y
    
    def as_tuple(self):
        out = (
            self.y_pk,
            self.x_pk,
            self.s_1,
            self.a_1,
            self.s_2,
            self.a_2,
            self.y_plat,   
        )
        return out
    

class VoutProfileFit:
    def __init__(self, xs: np.ndarray,
                 ys: np.ndarray):
        '''
        Here x is normalized by R_h, and y is normalized by V_h, 
        does not include Hubble flow.
        '''
        xs = np.array(xs)
        ys = np.array(ys)
        arg = ys.argmax()
        self.xs = xs
        self.ys = ys
        self.init_pf = FittedVoutProfile(
            y_pk = ys[arg],
            x_pk = xs[arg],
            s_1 = xs[arg],
            a_1 = .5,
            s_2 = .2,
            a_2 = 0.,            
            y_plat=.25,
        )
        
    def find_optim(self):
        param0 = self.init_pf.as_tuple()
        bounds = [
            (0., 5.),
            (0.001, 2.0),
            (0.001, 10.0),
            (-0.99, 5.0),
            (0.001, 10.0),
            (-0.99, 5.0),
            (-5.0, 5.0),
        ]
        out = minimize(self.f_obj, param0,
            method='Nelder-Mead', bounds=bounds,
            options={'maxiter':50000})
        if not out.success:
            print(out.message)
        self.opt_pf = FittedVoutProfile(*out.x)
        
    def f_obj(self, params):
        pf = FittedVoutProfile(*params)
        ys_pred = pf.y_at(self.xs)
        res = ys_pred - self.ys
        
        prior = 0.
        #prior = pf.a_1**2 + pf.a_2**2
        #if pf.x_pk < 0.:
        #    prior += (pf.x_pk/0.01)**2
        #if pf.s_1 < 0.:
        #    prior += (pf.s_1/0.01)**2
        #if pf.s_2 < 0.:
        #    prior += (pf.s_2/0.01)**2
        
        return (res*res).sum() + prior
    
    
@dataclass(frozen=True, slots=True)
class FittedVoutMap:
    vin_pf: FittedVinProfile
    vout_pf: FittedVoutProfile
    a_in: float
    s_in: float
    a_out: float
    s_out: float
    
    def y_at(self, theta: np.ndarray|float, x: np.ndarray|float):
        vin_pf, vout_pf = self.vin_pf, self.vout_pf
        a_in, s_in, a_out, s_out = (
            self.a_in, self.s_in, self.a_out, self.s_out)
        s_out, s_in = max(s_out, .1), max(s_in, .1)
         
        y_in = vin_pf.y_at(x)
        y_out = vout_pf.y_at(x) 
        
        is_scalar = np.isscalar(y_out)
        theta = np.asarray(theta)
        abstheta = np.abs(theta)
        exp_out = (np.pi/2 - abstheta)/s_out
        exp_in = abstheta / s_in
        
        aoutp1 = a_out + 1.
        ainp1 = a_in + 1.
        y = y_out * np.exp(- exp_out**aoutp1) - y_in * np.exp(- exp_in**ainp1)
        if is_scalar:
            y = y.item()
        return y

    def as_tuple(self, param_only=True):
        out = (
            self.a_in,
            self.s_in,
            self.a_out,
            self.s_out,
        )
        if not param_only:
            out = (self.vin_pf, self.vout_pf) + out
        return out
    

class VoutMapFit:
    def __init__(self, thetas: np.ndarray, xs: np.ndarray,
                 y_map: np.ndarray,
                 vin_pf: FittedVinProfile,
                 vout_pf: FittedVoutProfile,
                 wgt_map: np.ndarray|None = None):
        '''
        Here x is normalized by R_h.
        Theta [rad] is inclination angle.
        y_map is normalized by V_h, does not include Hubble flow.
        '''
        theta_map, x_map = np.meshgrid(thetas, xs, indexing='ij')
        if wgt_map is None:
            wgt_map = np.ones_like(y_map)
        else:
            wgt_map = np.array(wgt_map)
        
        self.thetas = np.array(thetas)
        self.xs = np.array(xs)
        self.theta_map = theta_map
        self.x_map = x_map
        self.y_map = np.array(y_map)
        self.wgt_map = wgt_map
        self.vin_pf = vin_pf
        self.vout_pf = vout_pf
        
        self.init_map = FittedVoutMap(
            vin_pf=vin_pf,
            vout_pf=vout_pf,
            a_in=1.,
            s_in=0.5,
            a_out=1.,
            s_out=0.5,
        )
        
    def find_optim(self):
        param0 = self.init_map.as_tuple()
        bounds = [
            (-0.99, 5.0),
            (0.01, np.pi),
            (-0.99, 5.0),
            (0.01, np.pi),
        ]
        out = minimize(self.f_obj, param0,
                       method='Nelder-Mead', bounds=bounds,
            options={'maxiter':50000})
        if not out.success:
            print(out.message)
        self.opt_map = FittedVoutMap(self.vin_pf, self.vout_pf, *out.x)
        
    def f_obj(self, params):
        wgt_map = self.wgt_map
        fit = FittedVoutMap(self.vin_pf, self.vout_pf, *params)
        y_map_pred = fit.y_at(self.theta_map, self.x_map)
        res = self.y_map - y_map_pred
        
        prior = 0.
        #prior = fit.a_in**2 + fit.a_out**2
        #if fit.s_in < 0.:
        #    prior += (fit.s_in/0.01)**2
        #if fit.s_out < 0.:
        #    prior += (fit.s_out/0.01)**2
        
        return (res*res*wgt_map).sum() + prior
    
@dataclass(frozen=True, slots=True)
class FittedVthetaMap:
    a_neg1: float
    a_neg2: float
    a_pos1: float
    a_pos2: float
    A_neg: float
    x_neg: float
    b_1neg: float
    b_2neg: float
    s_1neg: float
    s_2neg: float
    A_pos: float
    x_pos: float
    b_1pos: float
    b_2pos: float
    s_1pos: float
    s_2pos: float

    def y_at(self, theta: np.ndarray | float, x: np.ndarray | float):
        theta, x = np.asarray(theta), np.asarray(x)
        assert theta.shape == x.shape
        is_scalar = np.isscalar(theta)
        
        s_1 = max(self.s_1neg, 0.01)
        s_2 = max(self.s_2neg, 0.01)
        y_neg = self.A_neg * fn_double_S(x, self.x_neg, s_2, self.b_2neg, s_1, self.b_1neg)
        
        s_1 = max(self.s_1pos, 0.01)
        s_2 = max(self.s_2pos, 0.01)
        y_pos = self.A_pos * fn_double_S(x, self.x_pos, s_2, self.b_2pos, s_1, self.b_1pos)
        
        s_neg = self._powpow_normed(theta, self.a_neg1, self.a_neg2)
        s_pos = self._powpow_normed(theta, self.a_pos1, self.a_pos2)
        y = y_pos * s_pos - y_neg * s_neg

        if is_scalar:
            y = y.item()
        return y

    @staticmethod
    def _powpow(theta, a1, a2):
        abst = np.abs(theta)
        signt = np.sign(theta)
        out = signt * abst**(1. + a1) * (np.pi/2-abst)**(1. + a2)
        if np.isnan(out).any():
            print(f'{a1=}, {a2=}, {theta=}, {out=}')
        return out
    
    @staticmethod
    def _powpow_normed(theta, a1, a2):
        out = FittedVthetaMap._powpow(theta, a1, a2)
        theta_all = np.linspace(-np.pi/2., np.pi/2., 180)
        out_all = FittedVthetaMap._powpow(theta_all, a1, a2)
        return out / out_all.max().clip(1.0e-6)

    def as_tuple(self):
        out = (
            self.a_neg1,
            self.a_neg2,
            self.a_pos1,
            self.a_pos2,
            self.A_neg,
            self.x_neg,
            self.b_1neg,
            self.b_2neg,
            self.s_1neg,
            self.s_2neg,
            self.A_pos,
            self.x_pos,
            self.b_1pos,
            self.b_2pos,
            self.s_1pos,
            self.s_2pos,
        )
        return out


class VthetaMapFit:
    def __init__(self, thetas: np.ndarray, xs: np.ndarray,
                 y_map: np.ndarray, wgt_map: np.ndarray):
        '''
        Here x is normalized by R_h.
        Theta [rad] is inclination angle.
        y_map is normalized by V_h.
        '''
        theta_map, x_map = np.meshgrid(thetas, xs, indexing='ij')
        if wgt_map is None:
            wgt_map = np.ones_like(y_map)
        else:
            wgt_map = np.array(wgt_map)

        self.thetas = np.array(thetas)
        self.xs = np.array(xs)
        self.theta_map = theta_map
        self.x_map = x_map
        self.y_map = np.array(y_map)
        self.wgt_map = wgt_map

        sel = theta_map > 0.
        y_sel = self.y_map[sel]
        A_pos = abs(y_sel.max())
        A_neg = abs(y_sel.min())

        self.init_map = FittedVthetaMap(
            a_neg1=0.25,
            a_neg2=0.5,
            a_pos1=1.75,
            a_pos2=0.2,
            
            A_neg=A_neg,
            x_neg=2.,
            b_1neg=.5,
            b_2neg=.5,
            s_1neg=5.,
            s_2neg=0.5,
            
            A_pos=A_pos,
            x_pos=0.1,
            b_1pos=0.5,
            b_2pos=0.5,
            s_1pos=0.5,
            s_2pos=0.05,
        )
        self.A_neg_init = A_neg
        self.A_pos_init = A_pos

    def find_optim(self):
        param0 = self.init_map.as_tuple()
        bounds = [
            (-0.99, 5.),
            (-0.99, 5.),
            (-0.99, 5.),
            (-0.99, 5.),
            
            (0.0, 5.0),
            (0.05, 5.0),
            (0., 5.0),
            (0., 5.0),
            (0.01, 10.),
            (0.01, 10.),
            
            (0.0, 5.0),
            (0.01, 1.0),
            (0., 5.0),
            (0., 5.0),
            (0.01, 10.0),
            (0.01, 10.0),
        ]
        out = minimize(self.f_obj, param0, method='Nelder-Mead', 
                       bounds=bounds, options={'maxiter':50000})
        if not out.success:
            print(out.message)
        self.opt_map = FittedVthetaMap(*out.x)

    def f_obj(self, params):
        wgt_map = self.wgt_map
        fit = FittedVthetaMap(*params)
        y_map_pred = fit.y_at(self.theta_map, self.x_map)
        res = self.y_map - y_map_pred
        prior = 0.
        return (res*res*wgt_map).sum() + prior
    
class EdgeOnRecon:
    def __init__(self, bin_edges: tuple[np.ndarray,...],
        vout_map: FittedVoutMap,         
        vtheta_map: FittedVthetaMap,
        ecosys_info: EcosysInfo):
        
        x_es, y_es = bin_edges
        x_cs = 0.5 * (x_es[1:] + x_es[:-1])
        y_cs = 0.5 * (y_es[1:] + y_es[:-1])
        x_mcs, y_mcs = np.meshgrid(x_cs, y_cs)
        
        R_h, redshift, V_h, M_h, sim_info = (
            ecosys_info.R_h, ecosys_info.z, ecosys_info.V_h,
            ecosys_info.M_h, ecosys_info.sim_info)
        
        self.x_es = x_es
        self.y_es = y_es
        self.x_cs = x_cs
        self.y_cs = y_cs
        self.x_mcs = x_mcs
        self.y_mcs = y_mcs
        
        self.R_h = R_h
        self.M_h = M_h
        self.V_h = V_h
        self.redshift = redshift
        self.sim_info = sim_info
        
        self.vout_map = vout_map
        self.vtheta_map = vtheta_map
        
    def run(self):
        sim_info = self.sim_info
        cosm = sim_info.cosmology
        R_h, V_h = self.R_h, self.V_h
        Ez = cosm.big_hubble(self.redshift) / cosm.big_hubble0
        H = cosm.hubble * 100.0 * Ez / 1.0e3     # km/s/kpc
        
        x_mcs, y_mcs = self.x_mcs, self.y_mcs
        rs = np.hypot(x_mcs, y_mcs)
        xs = rs / self.R_h
        
        cos_incs = x_mcs / rs
        sin_incs = y_mcs / rs
        incs = np.arcsin(sin_incs)
        
        sign = np.sign(x_mcs)
        e_thetas = -sin_incs * sign, cos_incs * sign
        e_rs = cos_incs, sin_incs
        
        v_rs = self.vout_map.y_at(incs, xs) * V_h + rs * H
        v_thetas = self.vtheta_map.y_at(incs, xs) * V_h
        
        v_xs = v_rs * e_rs[0] + v_thetas * e_thetas[0]
        v_ys = v_rs * e_rs[1] + v_thetas * e_thetas[1]
        self.data = DataDict({
            'v_rs': v_rs,
            'v_xs': v_xs,
            'v_ys': v_ys,
            'rs': rs, 
            'incs': incs,
            'R_h': R_h,
            'V_h': V_h,
            'e_thetas': e_thetas,
            'e_rs': e_rs,
        })