import glob
import json
import os
import subprocess as sp
import time
import pandas as pd
import numpy as np
import argparse

"""Data collector

This script is used to collect data from the simulation runs. The main
functionalities are:

1. Combine multiple batchs' results into one dataset folder;
2. Gather the information of the simulation runs and report.
    - Total size
    - # and the list of folders merged
    - # of all runs
3. Check if # of all-data, toc, and logs are the same

The script is intended to collaborate with possible local ssh command
to run remotely.
"""

def execute(cmd, verbose=False):
    """Execute a command and return the output."""
    if verbose:
        print('  ', cmd)
    out = sp.getoutput(cmd)
    if out:
        print('  ', out)

def collect(tag,
            root='../BBR_test/ns-3.27',
            out_folder=None):
    """Collect results from all the folders include the tag and put into
    output folder.
    """
    os.chdir(root)
    root = os.getcwd()      # get the absolute path
    results = sp.getoutput(f'ls -d *{tag}*').split('\n')
    results = list(filter(lambda r: not 'merged' in r, results))
    print(f'= Found {len(results)} folders with tag {tag}')

    # Create output folder
    if out_folder is None:
        out_folder = f'merged_results_{tag}_' + time.strftime('%b-%d-%H:%M:%S')
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)
    os.chdir(out_folder)
    for folder in ['dats', 'logs', 'cfgs']:
        if not os.path.exists(folder):
            os.mkdir(folder)
    os.chdir(root)
    
    # Collect all the folders
    merge_runs = {}
    for result in results:
        print(f'\n= Copying from {result} ...')
        num = {}
        logs = sp.getoutput(f'ls {result}/logs/log_*').split('\n')
        runs = list(map(lambda x: x.split('_')[-1][:-4], logs))
        merge_runs[result] = runs
        num['run'] = len(runs)
        
        # copy csvs
        execute(f'cp {result}/*.csv {out_folder}')

        # copy dat, log
        for folder in ['dats', 'logs']:
            assert os.path.isdir(f'{result}/{folder}')
            if folder == 'dats':
                if glob.glob(f'{result}/{folder}/*') == []:
                # if not os.path.exists(f'{result}/{folder}/*'):
                    print(f' - No {folder}/* in {result}!')
                    num['all-data'], num['toc'] = 0, 0
                    continue
                num['all-data'] = len(sp.getoutput(f'ls {result}/dats/all-data_*').split('\n'))
                num['toc'] = len(sp.getoutput(f'ls {result}/dats/toc_*').split('\n'))
            else:
                if glob.glob(f'{result}/{folder}/*') == []:
                # if not os.path.exists(f'{result}/{folder}/*'):
                    print(f' - No {folder}/* in {result}!')
                    continue
            execute(f'cp {result}/{folder}/* {out_folder}/{folder}')
        
        # copy cfgs to the corresponding subfolders
        if not os.path.isdir(f'{result}/cfgs'):
            continue
        os.makedirs(f'{out_folder}/cfgs/{result}')
        execute(f'cp -r {result}/cfgs/* {out_folder}/cfgs/{result}')
        print(f'= {result} copied: {num["run"]} runs, {num["all-data"]} all-data, {num["toc"]} toc')

    # dump merge runs
    with open(f'{out_folder}/merge_runs.json', 'w') as f:
        json.dump(merge_runs, f, indent=4)
    print(f'\n= Merge runs dumped to {out_folder}/merge_runs.json')

    # check size
    print(f'\n= Folder size: {sp.getoutput(f"du -sh {out_folder}")}')
    print(f'= All data size: {sp.getoutput(f"du -sh {out_folder}/dats")}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--tag', type=str, required=True,
                        help='The tag of the folders to be merged')
    parser.add_argument('-r', '--root', type=str, default='../BBR_test/ns-3.27',
                        help='The root folder of the simulation runs')
    parser.add_argument('-o', '--out_folder', type=str, default=None,
                        help='The output folder of the merged results')
    args = parser.parse_args()
    collect(args.tag, args.root, args.out_folder)
