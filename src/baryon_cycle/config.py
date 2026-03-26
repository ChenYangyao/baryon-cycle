from __future__ import annotations
import os
from pyhipp_sims import sims
from pathlib import Path
from pyhipp import plot
from pyhipp.core import DataDict

plot.runtime_config.use_stylesheets('mathtext-it')

class ColorSets:
    dark2 = plot.ColorSeq.predefined('dark2').get_rgba()
    tab10 = plot.ColorSeq.predefined('tab10').get_rgba()
    set1 = plot.ColorSeq.predefined('set1').get_rgba()
    set2 = plot.ColorSeq.predefined('set2').get_rgba()
    
    def __getitem__(self, name):
        if isinstance(name, tuple):
            return tuple(self[n] for n in name) 
        return getattr(self, name)

c_sets = ColorSets()

class NamedColors:
    k               = 'black'
    r               = c_sets.set1[0] 
    red             = c_sets.set1[0]
    pink            = '#c284b3'
    b               = c_sets.set1[1] 
    blue            = c_sets.set1[1]
    p               = c_sets.dark2[2] 
    purple          = c_sets.dark2[2]
    ly              = c_sets.dark2[5] 
    lightyellow     = c_sets.dark2[5]
    y               = c_sets.dark2[6] 
    yellow          = c_sets.dark2[6]
    g               = c_sets.dark2[0] 
    green           = c_sets.dark2[0]
    lg              = c_sets.set1[2] 
    lightgreen      = c_sets.set1[2]
    grey            = c_sets.dark2[-1]
    orange          = c_sets.dark2[1] 
    lightorange     = c_sets.set1[4]
    
    def __getitem__(self, name):
        if isinstance(name, tuple):
            return tuple(self[n] for n in name) 
        return getattr(self, name)
    

cs_named = NamedColors()

class ProjPaths:
    proj_dir = Path(os.environ.get('MAHGIC_WORK_DIR', os.getcwd())).resolve() / 'workspaces/sims/galaxies/baryon_cycle'
    
    data_dir = proj_dir / 'data'
    figs_dir = data_dir / 'figs'
    sims_dir = data_dir / 'sims'
    obs_dir = data_dir / 'obs'
    tables_dir = data_dir / 'tables'
    
    def sim_dir_of(self, sim_info: sims.SimInfo) -> Path:
        return self.sims_dir / sim_info.name
    
    @staticmethod
    def save_fig(file_name: str, **savefig_kw):
        plot.savefig(ProjPaths.figs_dir / file_name, **savefig_kw)