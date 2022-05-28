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
        self.program = 'brite-for-cbtnk'
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
        not_number = False
        self.csv = None
        self.recompile = False
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
            elif arg == '-compile':
                self.recompile = True
            elif '-j' in arg:
                self.n_thread = int(arg[2:])
            elif read_program:
                self.program = arg
                read_program = False
            elif read_csv:
                self.csv = arg
                read_csv = False
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

        command = './waf --run "scratch/%s -mid=%s' % (self.program, self.mid)
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
            cmd = f'./waf --run "scratch/{self.program} -mid={self.mid}'
            suffix = '" > %s/log_debug_%s.txt 2>&1' % (os.path.join(self.res_path, 'logs'), self.mid)
            for col in df.columns:
                cmd += ' -%s=%s' % (col, row[col])
                run_id += f'_{col}={row[col]}'
            self.run_map[run_id] = self.mid
            cmd += suffix
            self._run_in_thread(cmd, run_id)

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
        if not os.path.exists(csv):
            csv = os.path.join('../../script/', csv)
        if csv :
            self.execute_arg_group(csv)
            self.csv = csv
        else:
            if self.mark_on:
                self.dfs(0, [], True)
            self.dfs(0, [])
        self.out.close()

        # build first in serial to avoid conflicts
        if self.recompile:
            os.system('CXXFLAGS="-Wall" ./waf configure --with-brite=../../BRITE --visualize')
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
        for id, mid in self.run_map.items():
            if self.mark_on:
                cmark = 'Co' if self.co_map[id] else 'NonCo'
            for prefix in ['all-data', 'queue', 'toc']:
                csv = f'MboxStatistics/{prefix}_{mid}*.csv'

                fdir = os.path.join(self.res_path, 'dats')
                os.system(f'cp {csv} {fdir}')
            
            # only for manual reference now
            # with open(os.path.join(fdir, 'content.txt'), 'a+') as f:
            #     csv1 = f'all-data_{mid}.csv'
            #     f.write(f'{id},{cmark},{csv1}\n')

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
        test_arg_group()
    else:
        # check argument, print help info, pass
        if len(sys.argv) < 3:
            print("Usage: python multiRun.py [-csv CSV] [-compile] [-crosson] [-markon] [-changedat] [-jN_THREAD]")
            print("     [-program PROGRAM_NAME] [-param1 MIN:STEP:MAX] [-param2 MIN:STEP:MAX] ...")
            print("     -crosson        include cross traffic.")
            print("     -markon         add Co/NonCo in front of subfolder name.")
            print("     -changedat      change mid in dat file name to run ID.")
            print("     -csv CSV        use CSV for simulation settings instead of cmd arguments.")
            print("     -compile        recompile the ns-3")
            exit(1)
        main()


