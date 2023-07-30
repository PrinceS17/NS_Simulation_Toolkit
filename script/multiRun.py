from concurrent.futures import thread
from matplotlib import pyplot as plt
import math, os, sys, time, random
import threading
from threading import Thread
import pandas as pd
import seaborn as sns
import time
import numpy as np
# from . import inflator
import inflator

is_test = False

class MultiRun_Module:
    ''' Batch simulation scanning general parameter within specified range.'''

    def __init__(self, folder=None, scpt=None, debug_time=None, config_path=None):
        self.path = ''       # path of current processing
        self.res_path = ''   # path of results: figs and logs
        self.scpt_path = ''  # path of script containing mPlotData.sh
        # self.program = 'brite-for-all' # program name that we want to run
        # self.program = 'brite-for-cbtnk'
        # self.program = 'cbtnk-dumbbell'
        self.program = 'cbtnk-extended-db'
        self.mid = random.randint(9900, 99999)         # unique mid for each run
        self.params = []     # list of parameters of mrun
        self.ranges = []     # list of tuples (min, max, step) corresponding to params above
        self.run_map = {}    # run_id -> mid, for later fetch of data
        self.co_map = {}     # run_id -> cobottleneck or not
        self.cross_on = False           # switch of drawing the cross traffic
        self.mark_on = False            # switch of mark Co/NonCo for each run
        self.change_dat_name = False    # if change dat name with run ID
        self.n_thread = 6
        self.threads = []
        self.debug_time = debug_time

        root = os.getcwd()
        root = root[:root.find ('Toolkit') + 7]
        self.path = folder or os.path.join(root, 'BBR_test', 'ns-3.27')
        self.config_path = config_path or os.path.join(self.path, 'edb_configs')
        self.scpt_path = scpt or os.path.join(root, 'script')
        print('root dir: {}'.format (root))
        print('ns path: %s' % self.path)
        print('scpt path: %s' % self.scpt_path)
        try:
            os.chdir(self.path)
        except:
            print('ns-3 not installed yet!')

        self.overwrite_inflation = False
        self.csv = None
        self.rebuild = False
        self.config_tag = []
        self.dry_run = False
        self.thread_duration = []
        self.use_monitor = False

    def _set_folder(self, tag, res_path=None):
        subdir = f'results_{tag}_' + time.strftime('%b-%d-%H:%M:%S') + \
            str(random.randint(0, 100))
        self.res_path = res_path or os.path.join(self.path, subdir)
        self.cfg_out_path = os.path.join(self.res_path, 'cfgs')
        os.makedirs(self.res_path, exist_ok=True)
        os.makedirs(os.path.join(self.res_path, 'logs'))
        self.out = open(os.path.join(self.res_path, 'logs', 'run_log.txt'), 'w')
        os.makedirs(os.path.join(self.res_path, 'figs'))
        os.makedirs(os.path.join(self.res_path, 'dats'))
        os.makedirs(self.cfg_out_path)

    def parse(self, args):
        ''' Read input arguments from bash. Args format: -cInt 0.02:0.02:0.08 -nProtocol 1:1:8.
        This part may seem outdated now, and the reasoning at first is to scale the input of
        all types of ns-3 args passed to multiRun, and directly feed them to each ns-3 run.
        Thus, instead of a few fixed field, all the other args should be directly transmitted
        to ns-3. As it grows larger and larger, the test overhead for transition to argparse
        becomes larger as well...

        TODO: improve using argparse
        '''
        read_program = False
        read_csv = False
        read_config_tag = False
        read_config_folder = False
        read_debug_time = False
        not_number = False
        read_res_path = False
        for arg in args:
            if arg == '-test':
                is_test = True
            elif arg == '-debug-time':
                read_debug_time = True
            elif arg == '-crosson':
                self.cross_on = True
            elif arg == '-crossoff':
                self.cross_on = False
            elif arg == '-markon':
                self.mark_on = True
            elif arg == '-changedat':
                self.change_dat_name = True
            elif arg == '-program':
                read_program = True
                continue
            elif arg == '-csv':
                read_csv = True
                continue
            elif arg == '-config-tag':
                read_config_tag = True
                continue
            elif arg == '-config-folder':
                print('check config-folder now')
                read_config_folder = True
            elif arg == '-res-path':
                read_res_path = True
            elif arg == '-rebuild':
                self.rebuild = True
            elif arg == '-overwrite-inflation':
                self.overwrite_inflation = True
            elif arg == '-dry-run':
                self.dry_run = True
            elif '-j' in arg:
                self.n_thread = int(arg[2:])
            elif arg == '-use-monitor':
                self.use_monitor = True
            elif read_debug_time:
                self.debug_time = list(map(float, arg.split(':')))
                assert len(self.debug_time) == 2
            elif read_program:
                self.program = arg
                read_program = False
            elif read_csv:
                self.csv = arg
                read_csv = False
            elif read_config_tag:
                self.config_tag.extend([arg, arg + '_spec'])
                read_config_tag = False
            elif read_config_folder:
                if '/' not in arg:
                    self.config_path = os.path.join(self.config_path, arg)
                else:
                    self.config_path = arg
                read_config_folder = False
            elif read_res_path:
                res_path = arg
                read_res_path = False
            elif arg[:2] == '--':       # used to specify non-number parameters
                self.params.append(arg[2:])
                not_number = True
            elif arg[0] == '-':
                self.params.append(arg[1:])
            elif len(self.params) - len(self.ranges) == 1:
                if not_number:
                    self.ranges.append((arg,))
                    not_number = False
                else:
                    th1, step, th2 = arg.split(':')
                    if self.params[-1] == 'tid' or self.params[-1] == 'mid':
                        self.ranges.append( (int(th1), int(step), int(th2)) )
                    else:    
                        self.ranges.append( (float(th1), float(step), float(th2)) )
            else:
                print('Error: parameters must be followed by a range!')
                exit(1)
        self._set_folder(self.config_tag[0], res_path)

    def mark(self, name, value):
        ''' Mark each run with Co or NonCo. '''
        val_str = [str(val) for val in value]
        run_id = '_'.join( ['='.join(c) for c in zip(name, val_str)] )
        self.run_map[run_id] = self.mid
        has_co = input('Does run ' + run_id + ' have a cobottleneck? (y/n) ')
        self.co_map[run_id] = True if has_co == 'y' else False
        return has_co

    def run_cmd(self, cmd, t_idx):
        ''' Mainly for threading. Note that thread_time doesn't work as the
        os.system() command duration is not recorded.
        '''
        t1 = time.time()
        t1_str = time.strftime('%b-%d-%H:%M:%S')
        print(f'  Thread {t_idx} starts at {t1_str} s')
        os.system(cmd)
        t2 = time.time()
        t2_str = time.strftime('%b-%d-%H:%M:%S')
        print(f'  Thread {t_idx} ends at {t2_str} s, duration: {t2 - t1:.3f} s')
        self.thread_duration[t_idx] = t2 - t1

    def execute(self, name, value):
        ''' Execute a parameter given its name and range in the form of (min, max, step). '''
        val_str = [str(val) for val in value]
        run_id = '_'.join( ['='.join(c) for c in zip(name, val_str)] )
        self.run_map[run_id] = self.mid

        command = f'./waf --run "scratch/{self.program} -{self.id_param}={self.mid}'
        for para, val in zip(name, value):
            command += ' -%s=%s' % (para, val)
        command += '" > %s/log_debug_%s.txt 2>&1' % (os.path.join(self.res_path, 'logs'), self.mid)
        self._run_in_thread(command, run_id)

    def execute_arg_group(self, csv):
        print(os.getcwd())
        df = pd.read_csv(csv, index_col=False, comment='#')
        cmds = []
        
        for _, row in df.iterrows():
            run_id = ''
            cmd = f'./waf --run "scratch/{self.program} -{self.id_param}={self.mid}'
            suffix = '" > %s/log_debug_%s.txt 2>&1' % (os.path.join(self.res_path, 'logs'), self.mid)
            for col in df.columns:
                cmd += ' -%s=%s' % (col, row[col])
                run_id += f'_{col}={row[col]}'
            self.run_map[run_id] = self.mid
            cmd += suffix
            self._run_in_thread(cmd, run_id)

    def _compile_configs(self):
        """Compiles configs, i.e. base / specific config csvs, to generate csvs
        for each ns-3 run.

        Returns config_dfs: [link_config_dfs, flow_config_dfs, cross_config_dfs],
                            each may contain different number of dfs based on inflation.
        """
        config_dfs = []
        for config_type in self.config_types:
            base_csv = os.path.join(self.config_path, f'{self.config_tag[0]}_{config_type}.csv')
            specific_csv = os.path.join(self.config_path, f'{self.config_tag[1]}_{config_type}.csv')
            # print('base_csv', base_csv, '\nspecific_csv', specific_csv)
            if config_type != 'wifi':
                assert os.path.exists(base_csv)
                df_base = pd.read_csv(base_csv, index_col=False, comment='#')
                assert 'run' not in df_base.columns, 'Switch base and specific csvs?'
                if not df_base.empty:
                    df_base = inflator.inflate_rows(df_base, is_test)
            else:
                # don't append any df, just skip wifi
                continue
            if os.path.exists(specific_csv):
                df_specific = pd.read_csv(specific_csv, index_col=False, comment='#')
            config_dfs.append([])
            if not os.path.exists(specific_csv) or df_specific.empty:   # specific csv is not necessary 
                config_dfs[-1].append(df_base)
                continue

            # inflate specific config csv
            f_inflated = f'{self.config_tag[0]}_{config_type}_inflated.csv'
            old_inflated_csv = os.path.join(self.config_path, f_inflated)
            new_inflated_csv = os.path.join(self.cfg_out_path, f_inflated)
            df_inflated = None

            def _write_inflated_config():
                df_inflated = inflator.inflate_rows(df_specific, is_test)
                df_inflated.to_csv(new_inflated_csv, index=False)
                print(f'Inflated config csv saved to {new_inflated_csv}')
                return df_inflated

            if not self.overwrite_inflation:
                # check out folder, but rarely there
                if os.path.exists(new_inflated_csv):
                    df_inflated = pd.read_csv(new_inflated_csv, index_col=False,
                                              comment='#')
                    print(f'Inflated config csv loaded from {new_inflated_csv}')
                elif os.path.exists(old_inflated_csv):
                    # copied from edb config if found there
                    df_inflated = pd.read_csv(old_inflated_csv, index_col=False,
                                              comment='#')
                    os.system(f'cp {old_inflated_csv} {self.cfg_out_path}')
                    print(f'Inflated config csv copied and loaded from {old_inflated_csv}')
                else:
                    print(f'No old inflated config found, rewriting it...')
                    df_inflated = _write_inflated_config()
            else:
                df_inflated = _write_inflated_config()

            # combine base df and specific run info into config df for each ns-3 run
            for run in df_inflated.run.unique():
                df = df_base.copy()
                df_inflated_run = df_inflated[df_inflated.run == run]
                tmp_df = pd.DataFrame(columns=df.columns)
                tmp_list = []
                for _, row in df_inflated_run.iterrows():
                    i_row = df.loc[(df.src == row.src) & (df.dst == row.dst)].index
                    new_row = row.drop('run')
                    if not i_row.empty:
                        df = df.drop(i_row)
                    assert (new_row.index == df.columns).all(), \
                        'Mismatched columns in spec and base configs!'
                    tmp_list.append(new_row.tolist()) 
                    # for col in df_inflated_run.columns:
                    #     if col == 'run':
                    #         continue
                    #     i_row = df.loc[(df.src == row.src) & (df.dst == row.dst), col]
                    #     df.iloc[i_row, col] = row[col]
                tmp_df = pd.DataFrame(tmp_list, columns=df.columns)
                df = pd.concat([df, tmp_df], ignore_index=True)
                df = df.sort_values(by=['src', 'dst'])
                config_dfs[-1].append(df)
        return config_dfs

    def execute_configs(self):
        """Execute all configs by parsing base and specific configs & generating
        tmp configs for ns-3.
        """
        self.config_types = ['link', 'flow', 'cross', 'wifi']
        config_dfs = self._compile_configs()
        if len(config_dfs) < 4:
            assert len(config_dfs) == 3
            self.config_types = self.config_types[:len(config_dfs)]
        tmps = os.path.join(self.cfg_out_path, 'tmps')
        if not os.path.exists(tmps):
            os.mkdir(tmps)
        os.chdir(tmps)

        # Specification loop: although using combinations which seems more general,
        # we use the zip method below to ensure a consistent meaning of 'run' in
        # all three files. Based on our experience, scenario can hardly be meaningful
        # w/o the careful choice of all link/flow/cross configs, so the combination
        # of these three may often waste runs, and binding them together is more
        # reasonable for our experiments.
        df_lens = [len(df) for df in config_dfs]
        if max(df_lens) != min(df_lens):
            for i in range(len(config_dfs)):
                if len(config_dfs[i]) < max(df_lens):
                    if len(config_dfs[i]) > 1:
                        print(f'\n=== Dangling inflation warning: {self.config_types[i]} config '
                           'is not fully specified, last run is used to complete the rest! ===\n')
                    config_dfs[i] += [config_dfs[i][-1]] * (max(df_lens) - len(config_dfs[i]))

        type_str = self.config_types
        for i, dfs in enumerate(zip(*config_dfs)):
            run_id = ''
            for j in range(len(type_str)):
                tag = f'{self.config_tag[0]}_{i}'
                dfs[j].to_csv(f'{tag}_{type_str[j]}.csv', index=False)
                run_id += f' -{type_str[j]}Config={tag}'
             # run ns3
            cmd = f'./waf --run "scratch/{self.program} -verbose=2 ' \
                  f'-useMonitor={self.use_monitor} ' \
                  f'-{self.id_param}={self.mid} -configFolder={tmps}'
            if self.debug_time is not None:
                cmd += f' -debugStart={self.debug_time[0]} -debugEnd={self.debug_time[1]}'
            suffix = '" > %s/log_debug_%s.txt 2>&1' % (os.path.join(self.res_path, 'logs'), self.mid)
            self.run_map[run_id] = self.mid
            cmd = cmd + run_id + suffix
            self._run_in_thread(cmd, run_id, i)

        self.tmps = tmps

    def _run_in_thread(self, command, run_id, t_idx=-1):
        self.thread_duration.append(0)
        if not is_test and not self.dry_run:
            print(f'\n  - Thread {len(self.threads)}: {command}')
            # os.system(command)
            t = Thread(target=self.run_cmd, args=(command, t_idx))
            self.threads.append(t)

        # print('    ', run_id, ' -> Run', self.mid)
        self.out.write(run_id + '\n')
        self.mid += 1

        return command

    def scan_all(self, csv=None):
        ''' Scan all the parameters input from command line using DFS.'''
        csv = csv if csv else self.csv
        self.id_param = 'mid'
        if self.program == 'cbtnk-dumbbell':
            self.id_param = 'run_id'
        elif self.program == 'cbtnk-extended-db':
            self.id_param = 'runId'
        if csv:
            if not os.path.exists(csv):
                csv = os.path.join('../../script/', csv)
            self.execute_arg_group(csv)
            self.csv = csv
        elif self.config_tag:
            self.execute_configs()
        else:
            if self.mark_on:
                self.dfs(0, [], True)
            self.dfs(0, [])
        self.out.close()
        os.chdir(self.path)

        if self.dry_run:
            print(f'\n - Dry run, no multithreading.')
            return

        # build first in serial to avoid conflicts
        if self.rebuild:
            os.system('CXXFLAGS="-Wall" ./waf configure --with-brite=../../BRITE --visualize > /dev/null')
            os.system('./waf build')

        # support multithreading
        t1 = time.time()
        print(f'\n - Begin multithreading with {self.n_thread} threads for {len(self.threads)} tasks in total')

        # use active_count to schedule a new thread ASAP to avoid wasting time!
        i = 0
        n_basic = threading.active_count()
        while i < len(self.threads):
            if threading.active_count() - n_basic < self.n_thread:
                self.threads[i].start()
                time.sleep(2.5)               # avoid reading the compile_commands.json at the same time
                i += 1
                interval = 0
            else:
                interval = 1
            time.sleep(interval)
        for t in self.threads:
            t.join()
        t2 = time.time()
        print(f'\n - Overall duration: {t1} -> {t2} = {t2 - t1:.3f}s')

    def dfs(self, index, value, flag=False):
        ''' DFS of scan_all: false for execute, true for mark. '''
        if index == len(self.params):
            if not flag:
                self.execute(self.params, list(value))
            else:
                self.mark(self.params, list(value))
            return
        
        if len(self.ranges[index]) == 1:        # non-number param
            value.append(self.ranges[index][0])
            self.dfs(index + 1, value, flag)
            value.pop()
        else:
            th1, step, th2 = self.ranges[index]
            num = int((th2 - th1) / step) + 1
            for i in range(num):
                value.append(th1 + i * step)
                self.dfs(index + 1, value, flag)
                value.pop()

    def plot_all(self, show_flow=None):
        if self.dry_run:
            return
        for id, mid in self.run_map.items():
            csv = f'MboxStatistics/all-data_{mid}.csv'
            df = pd.read_csv(csv, index_col=False)
            if show_flow:
                df = df[df.flow < show_flow]
            for field in df.columns:
                if field in ['time', 'flow']:
                    continue
                plt.figure()
                sns.lineplot(x='time', y=field, hue='flow', data=df)
                plt.savefig(os.path.join(self.res_path, 'figs', f'{field}_{mid}.pdf'))
                plt.close()
    
    def collect_all(self):
        if self.dry_run:
            return
        if self.csv:
            os.system(f'cp {self.csv} {self.res_path}')
        if self.config_tag:
            for tag in self.config_tag:
                # if overwrite-inflation is enabled, then the inflated csv under config_path
                # and the newly generated ones under config_path/cfgs will be different, thus
                # we need to overwrite the old one in the result folder
                os.system(f'cp {self.config_path}/{tag}_*.csv {self.res_path}')
                os.system(f'cp {self.cfg_out_path}/*.csv {self.res_path}')

        for id, mid in self.run_map.items():
            if self.mark_on:
                cmark = 'Co' if self.co_map[id] else 'NonCo'
            for prefix in ['all-data', 'queue', 'toc', 'rate']:
                csv = f'MboxStatistics/{prefix}_{mid}*.csv'

                fdir = os.path.join(self.res_path, 'dats')
                os.system(f'cp {csv} {fdir}')
                os.system(f'mv {csv} {csv}.bk')     # avoid mid collision in later runs

        # use cfg_out_path now, no need to mv
        # os.system(f'mv {self.tmps} {self.res_path}')


    def visualize(self, run_id):
        ''' Draw the result data rates given run_id, return mid for reference. '''
        # e.g. ~/scpt/mPlotData.sh -f lr_dbg -d DataRate_8803 -o y -s l -n 3'

        assert run_id in self.run_map, 'Run id not found!'
        mid = self.run_map[run_id]
        fig = 'Rate_' + run_id
        command = '%s/mPlotData.sh -p %s -f %s -d DataRate_%s -o n -s l -n 3' % (self.scpt_path, self.path, fig, mid)
        os.system(command)
        os.system('mv MboxFig/%s.png %s' % (fig + '*', os.path.join(self.res_path, 'figs')))

        if not self.cross_on:
            return mid, command

        cfig = 'CrossRate_' + run_id
        command = '%s/mPlotData.sh -p %s -f %s -d DataRate_%s -o n -s l -n 3 -m 3' % (self.scpt_path, self.path, cfig, mid)
        os.system(command)
        os.system('mv MboxFig/%s.png %s' % (cfig + '*', os.path.join(self.res_path, 'figs')))

        return mid, command

    def show_all(self):
        # deprecating
        ''' Draw all the result using run_map and put into figs directory. '''
        for id in self.run_map:
            self.visualize(id)
    
    def collect_all_old(self):
        # deprecating
        ''' Collect all the RTT and LLR data. '''
        os.chdir(os.path.join(self.res_path, 'dats'))
        for id in self.run_map:
            mid = self.run_map[id]
            subdir = 'mid=' + str(mid) + '_' + id
            if self.mark_on:
                cmark = 'Co' if self.co_map[id] else 'NonCo'
                subdir = cmark + '_mid=' + str(mid) + '_' + id
            os.mkdir(subdir)
            cp_cmd = 'cp ../../MboxStatistics/RttLlr_%s_*.dat ../../MboxStatistics/AckLatency_%s_*.dat %s' % (mid, mid, subdir)
            os.system(cp_cmd)
            tmp = os.popen('ls %s/RttLlr_%s_*.dat' % (subdir, mid)).read()
            N_flow = len(tmp.strip().split('\n'))
            os.chdir(subdir)
            for j in range(N_flow):
                if not self.change_dat_name:
                    break
                mv_cmd = 'mv RttLlr_%s_%s.dat RttLlr_%s_%s.dat' % (mid, j, id, j)
                mv_cmd2 = 'mv AckLatency_%s_%s.dat AckLatency_%s_%s.dat' % (mid, j, id, j)
                os.system(mv_cmd)
                os.system(mv_cmd2)
            os.chdir(os.path.join(self.res_path, 'dats'))

''' possible test case here '''
def test_parse():
    mr1 = MultiRun_Module()
    print(mr1.path)
    print(mr1.res_path)
    assert os.getcwd() == mr1.path      # maybe not correct

    args = ['-cInt', '0.02:0.02:0.04', '-wnd', '10:10:30']
    mr1.parse(args)
    assert mr1.params == ['cInt', 'wnd']
    assert mr1.ranges == [(0.02, 0.02, 0.04), (10, 10, 30)]
    
    print('  -- Test for parse passed.\n')

def test_execute():
    mr2 = MultiRun_Module()
    name = ['cInt', 'wnd']
    value = [0.02, 10]
    cmd = mr2.execute(name, value)
    print(cmd)

    res = input('  -> is the command above correct? ')
    if res == 'y':
        print('  -- Test for execute passed.\n')
    else:
        print('  -- Test failed.')
        exit(1)

def test_dfs():
    mr3 = MultiRun_Module()
    args = ['-cInt', '0.02:0.02:0.04', '-wnd', '10:10:30']
    mr3.parse(args)
    assert mr3.params == ['cInt', 'wnd']
    assert mr3.ranges == [(0.02, 0.02, 0.04), (10, 10, 30)]

    mr3.scan_all()
    print('  - run_log.txt: ')
    os.system('cat %s' % os.path.join(mr3.res_path, 'logs', 'run_log.txt'))
    res = input('  -> is the file above correct? ')
    if res == 'y':
        print('  -- Test for dfs passed.\n')
    else:
        print('  -- Test failed.')
        exit(1)

def test_visualize():
    mr4 = MultiRun_Module()
    args = ['-cInt', '0.02:0.02:0.04', '-wnd', '10:10:30', '-crosson']
    mr4.parse(args)
    mr4.scan_all()
    for id in mr4.run_map:
        _, cmd = mr4.visualize(id)
        print(cmd)
        break
    res = input('  -> is the visualization command above correct? ')
    if res == 'y':
        print('  -- Test for visualize passed.\n')
    else:
        print('  -- Test failed.')
        exit(1)

def test_root():
    mr5 = MultiRun_Module()
    res = input('  -> is the root directory above correct? ')
    if res == 'y':
        print('  -- Test for root path passed.\n')
    else:
        print('  -- Test failed.')
        exit(1)

def test_arg_group():
    mr = MultiRun_Module()
    csv = '../../script/test.csv'
    mr.scan_all(csv)
    res = input('  -> is the run correctly initiated? ')
    if 'y' in res:
        print('  -- Test for arg group passed.\n')
    else:
        print('  -- Test failed.')
        exit(1)

def test_config():
    mr = MultiRun_Module()
    mr.config_tag = ['test', 'test-spec']
    mr.id_param = 'runId'
    mr.config_path = os.path.join(mr.config_path, 'test')
    mr.overwrite_inflation = True
    mr.execute_configs()
    res = input(f'  -> are the configs correctly generated in {mr.config_path}/tmp? ')
    if 'y' in res:
        print('  -- Test for config passed.\n')
    else:
        print('  -- Test failed.')
        exit(1)

def main():
    # TODO: hack of the # flow to plot for now
    ''' Logical using order of the external API of the class. '''
    # mr = MultiRun_Module()
    # folder = os.path.join('/home', 'sapphire', 'Documents', 'ns3_BBR', 'ns-3.27')
    mr = MultiRun_Module()
    mr.parse(sys.argv[1:])
    print('\n -- Parsing complete. Start scanning ...\n')
    mr.scan_all()
    print('\n -- Scanning comoplete.\n')
    mr.collect_all()
    print('\n -- All data collected.\n')
    # mr.plot_all(2)
    # mr.show_all()
    # print(' -- All figures stored ...')
    

# Note: script will overwrite the data file
if __name__ == "__main__":

    if '-test' in sys.argv:
        is_test = True
        # test_parse()
        # test_execute()
        # test_dfs()
        # test_visualize()
        # test_root()
        # test_arg_group()
        test_config()
    elif len(sys.argv) == 1 and '-h' in sys.argv:
        print("""
        There are two types of inflation: Cartesian product ([1 2]) and static ({1 2}).
        The Cartesian product will generate all combinations of the inflated list from
        multiple fields, and give each inflated row a new run; while the static inflation
        will bind all the static fields and scan it only once (called static group), and
        each static group has the same run No, e.g. {1 2} {3 4} in one row will generate
        two rows of the same run No., each with 1 3 and 2 4. The most common use case
        for static inflation is to set multiple (src, dst) pairs within one run.

        Take Cartesian product inflation as example, the grammar includes:
            - [start:step:end]: inflates to range(start, end, step)
            - [start-end]: inflates to range(start, end, 1)
            - [start]: inflates to [start]
            - [a b]: inflates to [a, b]
            - *: allows in run field, means automatically generate run No. based on static field row
        Replace [] with {} for static inflation.

        Also, it supports three types of distributions:
            N(mean std): positive samples drawn from N(mean, std);
            U(start end): unifrom distribution;
            C(start end step): choice drawn from [start:end:step].

        See inflate_rows() in this script and edb_configs/test for more details.
        """)
    elif len(sys.argv) < 3:
        print("Usage: python multiRun.py [-config-tag base_tag spec_tag] [-config-folder subfolder]")
        print("                          [-rebuild] [-overwrite-inflation] [-jN_THREAD]")
        print("                          [-csv CSV] [-crosson] [-markon] [-changedat]")
        print("                          [-debug-time start:end] [-res-path path]")
        print("     [-program PROGRAM_NAME] [-param1 MIN:STEP:MAX] [-param2 MIN:STEP:MAX] ...")
        print("     -test        run test cases.")
        print("     -crosson        include cross traffic.")
        print("     -markon         add Co/NonCo in front of subfolder name.")
        print("     -changedat      change mid in dat file name to run ID.")
        print("     -csv CSV        use CSV for simulation settings instead of cmd arguments.")
        print("     -config-tag base_csv specific_csv   base & specific csv for extended dumbbell simulation scan.")
        print("     -config-folder path or subfolder    subfolder under edb_configs for config files.")
        print("                     if provided one folder, then regarded as subfolder under edb_configs")
        print("                     else, regarded as the whole config path")
        print("     -res-path path    path to store the results, default is ns-3.27/result_xxx.")
        print("     -rebuild        compile and build the ns-3, necessary for avoiding collision among threads")
        print("     -overwrite-inflation   overwrite inflation even when previous inflated csv exists.")
        print("                     note that this only rewrites inflation in the output result dir but not edb configs")
        print("     -debug-time     the debug start & end time for ns-3, overwrite config's start/end time") 
        print("     -dry-run        do all things except running ns-3.")
        print("\n  For cbtnk-extended-db simulation, config csvs in config-folder\n"
        "  is supported. Typically link, flow, cross config csvs are supported, and\n"
        "  each type consists of base and specific csvs. The specific csvs will be\n"
        "  inflated to multiple runs and generate one config w/ base csv for each run.\n"
        "  A typical commands for cbtnk-extended-db simulation is:\n\n"
        
        "  python3 multiRun.py -config-folder new_cross -config-tag ppbp ppbp_spec\n"
        "     -rebuild -j6")
    else:
        main()
