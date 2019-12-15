from matplotlib import pyplot as plt
import os, sys, time, random

is_test = False

class MultiRun_Module:
    ''' Batch simulation scanning general parameter within specified range.'''

    def __init__(self, folder=None, scpt=None):
        self.path = ''       # path of current processing
        self.res_path = ''   # path of results: figs and logs
        self.scpt_path = ''  # path of script containing mPlotData.sh
        self.program = 'brite-for-all' # program name that we want to run
        self.mid = random.randint(11, 999)         # unique mid for each run
        self.params = []     # list of parameters of mrun
        self.ranges = []     # list of tuples (min, max, step) corresponding to params above
        self.run_map = {}    # run_id -> mid, for later fetch of data
        self.co_map = {}     # run_id -> cobottleneck or not
        self.cross_on = False           # switch of drawing the cross traffic
        self.mark_on = False            # switch of mark Co/NonCo for each run
        self.change_dat_name = False    # if change dat name with run ID

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
            elif read_program:
                self.program = arg
                read_program = False
            elif arg[0] == '-':
                self.params.append(arg[1:])
            elif len(self.params) - len(self.ranges) == 1:
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

    def execute(self, name, value):
        ''' Execute a parameter given its name and range in the form of (min, max, step). '''
        val_str = [str(val) for val in value]
        run_id = '_'.join( ['='.join(c) for c in zip(name, val_str)] )
        self.run_map[run_id] = self.mid

        command = './waf --run "scratch/%s -mid=%s' % (self.program, self.mid)
        for para, val in zip(name, value):
            command += ' -%s=%s' % (para, val)
        command += '" > %s/log_debug_%s.txt 2>&1' % (os.path.join(self.res_path, 'logs'), self.mid)

        if not is_test:
            os.system(command)

        print(run_id, ' -> Run', self.mid)
        self.out.write(run_id + '\n')
        self.mid += 1

        return command
    
    def scan_all(self):
        ''' Scan all the parameters input from command line using DFS.'''
        if self.mark_on:
            self.dfs(0, [], True)
        self.dfs(0, [])
        self.out.close()

    def dfs(self, index, value, flag=False):
        ''' DFS of scan_all: false for execute, true for mark. '''
        if index == len(self.params):
            if not flag:
                self.execute(self.params, list(value))
            else:
                self.mark(self.params, list(value))
            return
        
        th1, step, th2 = self.ranges[index]
        num = int((th2 - th1) / step) + 1
        for i in range(num):
            value.append(th1 + i * step)
            self.dfs(index + 1, value, flag)
            value.pop()
        

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
        ''' Draw all the result using run_map and put into figs directory. '''
        for id in self.run_map:
            self.visualize(id)
    
    def collect_all(self):
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


def main():
    ''' Logical using order of the external API of the class. '''
    # mr = MultiRun_Module()
    # folder = os.path.join('/home', 'sapphire', 'Documents', 'ns3_BBR', 'ns-3.27')
    mr = MultiRun_Module()
    mr.parse(sys.argv[1:])
    print(' -- Parsing complete. Start scanning ...')
    mr.scan_all()
    print(' -- Scanning comoplete. Start generating figures ...')
    mr.show_all()
    print(' -- All figures stored ...')
    mr.collect_all()
    print(' -- All Rtt & Llr data collected.')


# Note: script will overwrite the data fileg
if __name__ == "__main__":
    is_test = False     # test mode will disable the mrun command

    if is_test:         # test cases here: intended tests all passed
        # test_parse()
        # test_execute()
        # test_dfs()
        # test_visualize()
        test_root()
    else:
        # check argument, print help info, pass
        if len(sys.argv) < 3:
            print("Usage: python multiRun.py [-crosson] [-markon] [-changedat]")
            print("     [-program PROGRAM_NAME] [-param1 MIN:STEP:MAX] [-param2 MIN:STEP:MAX] ...")
            print("     -crosson        include cross traffic.")
            print("     -markon         add Co/NonCo in front of subfolder name.")
            print("     -changedat      change mid in dat file name to run ID.")
            exit(1)
        main()


