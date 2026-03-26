import os, baryon_cycle

def test_env():
    print('Python PATH:', os.environ.get('PYTHONPATH', ''))    
    
def test_version():
    print(baryon_cycle)
    
def test_proj_paths():
    print('Project dir:', baryon_cycle.ProjPaths.proj_dir)