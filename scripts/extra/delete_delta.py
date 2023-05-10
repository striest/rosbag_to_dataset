import os
from os.path import join, isdir
fp = '/project/learningphysics/tartandrive_trajs'

for traj in os.listdir(fp):
    delta_fp = join(fp,traj,'delta')
    if isdir(delta_fp):
        try: 
            os.system(f"rm -rf {delta_fp} ")
        except:
            print(f"{delta_fp} couldn't be removed")