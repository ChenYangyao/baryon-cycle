from __future__ import annotations
import typing
from typing import Self
from dataclasses import dataclass
from pyhipp_sims import sims
import numpy as np
from pyhipp.core import DataDict

@dataclass(frozen=True, slots=True)
class EcosysInfo:
    R_h: float                  # kpc, physical
    V_h: float                  # km/s, physical 
    M_h: float                  # Msun
    H: float                    # km/s/kpc
    z: float
    R_ISM: float
    R_CGM: float 
    sim_info: sims.SimInfo
    N_gal: int

    @staticmethod
    def from_dict(ecosys_def: dict[str, float|int]):
        R_h, z, sim_name = ecosys_def['R_h', 'z', 'sim_name']
        scale_a = 1. / (1. + z)
        R_h_c = R_h / scale_a                 # kpc
        Vol_c = (4./3.) * np.pi * R_h_c**3    # kpc^3
        
        sim_info = sims.predefined[sim_name]    
        us = sim_info.unit_system
        u_m = us.u_m_to_sol
        u_l = us.u_l_to_pc / 1.0e3
        u_rho = u_m / u_l**3
        u_vel = us.u_v_to_kmps
        ht = sim_info.cosmology.halo_theory
        rho = ht.rho_vir_crit(z=z) * u_rho                
        M_h = rho * Vol_c                                       # Msun
        V_h = ht.vir_props_crit(M_h/u_m, z=z).v * u_vel         # km/s, physical
        
        cosm = sim_info.cosmology
        Ez = cosm.big_hubble(z) / cosm.big_hubble0
        H = cosm.hubble * 100.0 * Ez / 1.0e3                    # km/s/kpc
        
        return EcosysInfo(
            R_h=R_h,
            V_h=V_h,
            M_h=M_h,
            H=H,
            z=z,
            R_ISM=ecosys_def['R_ISM'],
            R_CGM=ecosys_def['R_CGM'],
            sim_info=sim_info,
            N_gal=ecosys_def['N_gal']
        )
            
    def to_dict(self) -> DataDict[str, float|int|str]:
        return DataDict({
            'R_h': self.R_h,
            'V_h': self.V_h,
            'M_h': self.M_h,
            'H': self.H,
            'z': self.z,
            'R_ISM': self.R_ISM,
            'R_CGM': self.R_CGM,
            'sim_name': self.sim_info.name,
            'N_gal': self.N_gal,   
        })
    
