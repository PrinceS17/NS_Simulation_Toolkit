from concurrent.futures import thread
from matplotlib import pyplot as plt
import math, os, sys, time, random
import threading
from threading import Thread
import pandas as pd
import seaborn as sns
import time

is_test = False

class MultiRun_Module:
    ''' Batch simulation scanning general parameter within specified range.'''

    def __init__(self, folder=None, scpt=None):
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

        root = os.getcwd()
        root = root[:root.find ('Toolkit') + 7]
        self.path = os.path.join(root, 'BBR_test', 'ns-3.27') if not folder else folder
        self.config_path = os.path.join(self.path, 'edb_configs')
        self.scpt_path = os.path.join(root, 'script') if not scpt else scpt
        print('root dir: {}'.format (root))
        print('ns path: %s' % self.path)
        print('scpt path: %s' % self.scpt_path)
        try:
            os.chdir(self.path)
        except:
            print('ns-3 not installed yet!')

        subdir = 'results_' + time.strftime('%b-%d-%H:%M:%S') + str(random.randint(0, 100))
        os.mkdir(subdir)
        os.mkdir(os.path.join(subdir, 'logs'))
        self.out = open(os.path.join(subdir, 'logs', 'run_log.txt'), 'w')
        os.mkdir(os.path.join(subdir, 'figs'))
        os.mkdir(os.path.join(subdir, 'dats'))
        self.res_path = os.path.join(self.path, subdir)
    
    def parse(self, args):
        ''' Read input arguments from bash. Args format: -cInt 0.02:0.02:0.08 -nProtocol 1:1:8. '''
        read_program = False
        read_csv = False
        read_config_tag = 0
        self.config_tag = []
        not_number = False
        self.csv = None
        self.rebuild = False
        for arg in args:
            if arg == '-crosson':
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
                read_config_tag = 1
                continue
            elif arg == '-rebuild':
                self.rebuild = True
            elif '-j' in arg:
                self.n_thread = int(arg[2:])
            elif read_program:
                self.program = arg
                read_program = False
            elif read_csv:
                self.csv = arg
                read_csv = False
            elif read_config_tag > 0:
                self.config_tag.append(arg)
                read_config_tag += 1
                if read_config_tag > 2:
                    read_config_tag = 0
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

    def mark(self, name, value):
        ''' Mark each run with Co or NonCo. '''
        val_str = [str(val) for val in value]
        run_id = '_'.join( ['='.join(c) for c in zip(name, val_str)] )
        self.run_map[run_id] = self.mid
        has_co = input('Does run ' + run_id + ' have a cobottleneck? (y/n) ')
        self.co_map[run_id] = True if has_co == 'y' else False
        return has_co

    def run_cmd(self, cmd):
        ''' Mainly for threading.'''
        os.system(cmd)

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
        df = pd.read_csv(csv, index_col=False)
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

    def execute_configs(self):
# TODO: consider link/flow/cross! input tag but not the exact csv name!
#       use base_tag, specific tag, for all the csvs
#       construct all threee csvs before running the cmd, as only one tag can be passed

        config_dfs = []
        # parse base & specific csv to merge the config dfs for all types
        for config_type in ['link', 'flow', 'cross']:
            config_dfs.append([])
            base_csv = os.path.join(self.config_path, f'{self.config_tag[0]}_{config_type}.csv')
            specific_csv = os.path.join(self.config_path, f'{self.config_tag[1]}_{config_type}.csv')
            assert os.path.exists(base_csv)
            df_base = pd.read_csv(base_csv, index_col=False)
            if os.path.exists(specific_csv):
                df_specific = pd.read_csv(specific_csv, index_col=False)
            if not os.path.exists(specific_csv) or df_specific.empty:   # specific csv is not necessary 
                config_dfs[-1].append(df_base)
                continue

            for run in df_specific.run.unique():
                df = df_base.copy()
                df_specific_run = df_specific[df_specific.run == run]
                for _, row in df_specific_run.iterrows():
                    for col in df_specific_run.columns:
                        if col == 'run':
                            continue
                        df.loc[(df.src == row.src) & (df.dst == row.dst), col] = row[col]
                config_dfs[-1].append(df)

        # scan the types to construct the final csv w/ single tag each ns-3 run
        tmps = os.path.join(self.config_path, 'tmps')
        if not os.path.exists(tmps):
            os.mkdir(tmps)
        os.chdir(tmps)
        for i, link_df in enumerate(config_dfs[0]):
            link_tag = f'{self.config_tag[0]}_{i}'
            link_df.to_csv(f'{link_tag}_link.csv', index=False)
            for j, flow_df in enumerate(config_dfs[1]):
                flow_tag = f'{self.config_tag[0]}_{j}'
                flow_df.to_csv(f'{flow_tag}_flow.csv', index=False)
                for k, cross_df in enumerate(config_dfs[2]):
                    cross_tag = f'{self.config_tag[0]}_{k}'
                    cross_df.to_csv(f'{cross_tag}_cross.csv', index=False)

                    # run ns3
                    cmd = f'./waf --run "scratch/{self.program} -{self.id_param}={self.mid} -configFolder={tmps}'
                    run_id = f' -linkConfig={link_tag} -flowConfig={flow_tag} -crossConfig={cross_tag}'
                    suffix = '" > %s/log_debug_%s.txt 2>&1' % (os.path.join(self.res_path, 'logs'), self.mid)
                    self.run_map[run_id] = self.mid
                    cmd = cmd + run_id + suffix
                    self._run_in_thread(cmd, run_id)
        self.tmps = tmps

    def _run_in_thread(self, command, run_id):
        if not is_test:
            print(f'  - Thread {len(self.threads)}: {command}')
            # os.system(command)
            t = Thread(target=self.run_cmd, args=(command,))
            self.threads.append(t)

        print('    ', run_id, ' -> Run', self.mid)
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

        # build first in serial to avoid conflicts
        if self.rebuild:
            os.system('CXXFLAGS="-Wall" ./waf configure --with-brite=../../BRITE --visualize > /dev/null')
            os.system('./waf build')

        # support multithreading
        t1 = time.time()
        print(f'\n - Begin multithreading with {self.n_thread} threads for {len(self.threads)} tasks in total')
        # for i in range(math.ceil( len(self.threads) / self.n_thread)):
        #     for mode in [0, 1]:
        #         for j in range(self.n_thread):
        #             n = i * self.n_thread + j
        #             if n == len(self.threads):
        #                 break
        #             if not mode:
        #                 self.threads[n].start()
        #                 print(f'    Starting thread {n}')
        #             else:
        #                 self.threads[n].join()

        # use active_count to schedule a new thread ASAP to avoid wasting time!
        i = 0
        n_basic = threading.active_count()
        while i < len(self.threads):
            if threading.active_count() - n_basic < self.n_thread:
                self.threads[i].start()
                print(f'    Starting thread {i}')
                i += 1
                interval = 0
            else:
                interval = 1
            time.sleep(interval)
        for t in self.threads:
            t.join()
        t2 = time.time()
        print(f'\n - Duration: {t1} -> {t2} = {t2 - t1:.3f}s')

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
        if self.csv:
            os.system(f'cp {self.csv} {self.res_path}')
        if self.config_tag:
            for tag in self.config_tag:
                os.system(f'cp {self.config_path}/{tag}_*.csv {self.res_path}')

        for id, mid in self.run_map.items():
            if self.mark_on:
                cmark = 'Co' if self.co_map[id] else 'NonCo'
            for prefix in ['all-data', 'queue', 'toc', 'rate']:
                csv = f'MboxStatistics/{prefix}_{mid}*.csv'

                fdir = os.path.join(self.res_path, 'dats')
                os.system(f'cp {csv} {fdir}')

        # os.system(f'rm -r {self.tmp}')


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
    print(' -- Parsing complete. Start scanning ...')
    mr.scan_all()
    print(' -- Scanning comoplete.')
    mr.collect_all()
    print(' -- All data collected.')
    mr.plot_all(2)

    # mr.show_all()
    # print(' -- All figures stored ...')
    


# Note: script will overwrite the data file
if __name__ == "__main__":
    is_test = False     # test mode will disable the mrun command

    if is_test:         # test cases here: intended tests all passed
        # test_parse()
        # test_execute()
        # test_dfs()
        # test_visualize()
        # test_root()
        # test_arg_group()
        test_config()
    else:
        # check argument, print help info, pass
        if len(sys.argv) < 3:
            print("Usage: python multiRun.py [-csv CSV] [-config-tag base_csv specific_csv] [-rebuild] [-crosson] [-markon] [-changedat] [-jN_THREAD]")
            print("     [-program PROGRAM_NAME] [-param1 MIN:STEP:MAX] [-param2 MIN:STEP:MAX] ...")
            print("     -crosson        include cross traffic.")
            print("     -markon         add Co/NonCo in front of subfolder name.")
            print("     -changedat      change mid in dat file name to run ID.")
            print("     -csv CSV        use CSV for simulation settings instead of cmd arguments.")
            print("     -config-tag base_csv specific_csv   base & specific csv for extended dumbbell simulation scan.")
            print("     -rebuild        compile and build the ns-3, necessary for avoiding collision among threads")
            exit(1)
        main()


