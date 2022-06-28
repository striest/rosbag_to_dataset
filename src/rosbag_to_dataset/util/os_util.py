"""
Collection of functions for manipulating files, saving data, etc.
"""

import os
import argparse
import shutil

def maybe_mkdir(fp, force=True):
    if not os.path.exists(fp) or force:
        os.makedirs(fp)
    # elif not force:
    #     x = input('{} already exists. Hit enter to continue and overwrite. Q to exit.'.format(fp))
    #     if x.lower() == 'q':
    #         exit(0)

def maybe_rmdir(fp, force=True):
    if os.path.exists(fp) and force:
        shutil.rmtree(fp)
    elif not force:
        try:
            os.rmdir(fp)
        except OSError as error:
            x = input('{} is not empty. Hit "r" to continue and remove.'.format(fp))
            if x.lower() == 'r':
                shutil.rmtree(fp)

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
