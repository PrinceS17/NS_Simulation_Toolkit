import argparse
import numpy as np
import os
import pandas as pd
import unittest

class ConfigGenerator:
    """Config generator. This is used to generate large scale simulation config csvs
    in a universal way with the support of inflation and base/specific dfs structure
    from multiRun. The architecture is like:

        1 Experiment set: 1 folder of all configs;
        -> several simulation groups based on fixed bottleneck link number;
        -> several link settings (leaf, btnk, mid, etc)

    The bindings across configs include:
        1. Run number. Flow and cross traffic must be generated according to the right
        run, otherwise the src/dst set is wrong.
        2. src/dst number. This is used in generating flow number.
        3. Simulation start and end time. It's used both in flow and cross configs.
    
    Simulation group = {
        'runs' :[1,2,...],
        'src_to_dst': {1: [2,3], 2: [4,5], ...},
        'leaf_gw': {2: 1, 0: 4, ...},
        'leaf_queue': {2: 0, 0: 1, ...},
        'sim_start': 0.0,
        'sim_end': 10 * 60,
    }

    Low level grammar style should be: using distribution extensively, but generally
    don't inflate fields other than run for clear visualization in csv.
    """

    def __init__(self, folder, tag, root='../BBR_test/ns-3.27/edb_configs'):
        """Maintains the dir and csv for output.
        """
        self.folder = os.path.join(root, folder)
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)
        self.tag = tag
        self.next_run = 0
        self.col_map = {
            'link': ['src', 'dst', 'position', 'type', 'bw_mbps', 'delay_ms',
                     'q_size', 'q_type', 'q_monitor'],
            'flow': ['src', 'dst', 'src_gw', 'dst_gw', 'num', 'rate_mbps', 'delayed_ack',
                    'start', 'end', 'q_index1', 'q_index2'],
            'cross': ['src', 'dst', 'num', 'mode', 'cross_rate_mbps', 'edge_rate_mbps',
                    'hurst', 'mean_duration', 'start', 'end'],
        }
        self.data = {}
        for typ, cols in self.col_map.items():
            # self.data[typ] = {k: [] for k in ['run'] + cols}
            self.data[typ] = []

    def init_group(self, n_btnk, n_run=10, sim_start=0, sim_end=600):
        self.group = {
            'n_btnk': n_btnk,
            'runs': [self.next_run, self.next_run + n_run - 1],     # inclusive
            'run_str': f'[{self.next_run}-{self.next_run + n_run - 1}]' if n_run > 1 \
                else str(self.next_run),
            'leaves': [[], []],     # [src nodes, dst nodes]
            'leaf_gw': {},
            'gw_mid': [[], []],
            'gw_queue': {},         # gw -> queue index for flow config
            'sim_start': sim_start,
            'sim_end': sim_end,
            'next_node': 0,         # the next node id to use for the current group
            'next_queue': 0,
        }
        self.next_run += n_run

    def _add_gw_queue(self, gw):
        """Atomic operation to add queue to gw_queue and update next queue."""
        self.group['gw_queue'][gw] = self.group['next_queue']
        self.group['next_queue'] += 1

    def _alloc_nodes(self, n):
        """Allocate n nodes for the group. Returns [start, end]."""
        start, end = self.group['next_node'], self.group['next_node'] + n - 1
        self.group['next_node'] += n
        return start, end

    def output_csv(self, typ, is_spec):
        mid = 'spec_' if is_spec else ''
        csv = os.path.join(self.folder, f'{self.tag}_{mid}{typ}.csv')
        if not is_spec:
            df = pd.DataFrame(columns=self.col_map[typ])
        else:
            df = pd.DataFrame(self.data[typ], columns=['run'] + self.col_map[typ])
        df.to_csv(csv, index=False)
        print(f'Output csv: {csv}')

    def generate_link(self, n_leaf=None):
        """Generate link config csv based on the number of bottleneck links.
        Link config columns: 'src', 'dst', 'position', 'type', 'bw_mbps',
            'delay_ms', 'q_size', 'q_type', 'q_monitor'.

        Link structure: left / right leaf, left / right btnk, middle links
        Nodes structure: leaf -> gw -> mid left -> mid right -> gw -> leaf
        """
        max_leaf = [4, 7]           # server side smaller
        run_str = self.group['run_str']
        small_delay_str, large_delay_str = 'N(0.5 0.1)', 'L(2.9 1.3225)'
        small_bw_str, large_bw_str = 'C(100:100:1001)', '2000'
        qtype_str = 'C(pie codel)'
        qsize_str = 'C(100:100:1001)'              # TODO: queue's distribution
        mids = [[], []]     # [left_mids, right_mids]
        cur_link_data = []
        for i in range(self.group['n_btnk']):
            # TODO: this assumes n_left_btnk == n_right_btnk
            #       extension: support != cases, and not loop side over each btnk

            for side in range(2):       # left / right
                # leaf -> gw
                n_leaf_to_use = n_leaf
                if n_leaf is None:
                    n_leaf_to_use = np.random.choice(range(2, max_leaf[side]))
                leaf0, leaf1 = self._alloc_nodes(n_leaf_to_use)
                gw, _ = self._alloc_nodes(1)
                row = [run_str, f'[{leaf0}-{leaf1}]', gw, 'leaf', 'ppp',
                    large_bw_str, small_delay_str, qsize_str, qtype_str, 'none']
                cur_link_data.append(row)
                self.group['leaves'][side].extend(range(leaf0, leaf1 + 1))
                for leaf in range(leaf0, leaf1 + 1):
                    self.group['leaf_gw'][leaf] = gw

                # gw -> mid
                mid, _ = self._alloc_nodes(1)
                mids[side].append(mid)
                pos = 'left_mid' if side == 0 else 'right_mid'
                link = [gw, mid] if side == 0 else [mid, gw] 
                row = [run_str, link[0], link[1], pos, 'ppp', small_bw_str,
                       small_delay_str, qsize_str, qtype_str, 'tx']
                cur_link_data.append(row)
                self.group['gw_mid'][side].append(gw)   # ensure the correct traffic direction
                self.group['gw_mid'][1 - side].append(mid)
                self._add_gw_queue(gw)

        # mid left -> mid right
        left_mids = '[' + ' '.join(map(str, mids[0])) + ']'
        right_mids = '[' + ' '.join(map(str, mids[1])) + ']'
        row = [run_str, left_mids, right_mids, 'mid', 'ppp', large_bw_str,
                large_delay_str, qsize_str, qtype_str, 'none']
        cur_link_data.append(row)

        self.data['link'].extend(cur_link_data)
        res_df = pd.DataFrame(cur_link_data, columns=['run'] + self.col_map['link']) # for test
        return res_df

    def generate_flow(self, dynamic_ratio=0.33):
        """Generate flow configs.

        Flow config: 'src', 'dst', 'src_gw', 'dst_gw', 'num', 'rate_mbps',
                    'delayed_ack', 'start', 'end', 'q_index1', 'q_index2'.

        Args:
            dynamic_ratio (float, optional): the ratio of dynamic window to total
                                             window size for flow start and end.
        """
        start, end = self.group['sim_start'], self.group['sim_end']
        dynamic_window = (end - start) * dynamic_ratio
        assert start >= 0 and end - dynamic_window >= 0
        run_str = self.group['run_str']
        rate_str = 'L(3 1.3225)'
        start_str = f'U({start} {start + dynamic_window})'
        end_str = f'U({end - dynamic_window} {end})'
        # From the literature, an edge src server has ~ 80 users at most.
        # We divide it by the number of dst to compute the number of users for each src
        # server, and obtain the power law for user number here.
        # (Note that here ns-3 path is assumed to contain 'num' users.)
        # TODO: the understanding of the figure seems wrong
        #       80 users have 1 servers doesn't mean 1 servers only have 80 users!
        #       80 is too less for an edge server, set to 200 for now
        user_per_dst = 200 / len(self.group["leaves"][1])
        num_str = f'P({user_per_dst} 0.44)'
        cur_flow_data = []
        for side in range(2):
            assert len(self.group['leaves'][side]) > 0
        for src in self.group['leaves'][0]:
            for dst in self.group['leaves'][1]:
                src_gw, dst_gw = self.group['leaf_gw'][src], self.group['leaf_gw'][dst]
                q1, q2 = self.group['gw_queue'][src_gw], self.group['gw_queue'][dst_gw]
                row = [run_str, src, dst, src_gw, dst_gw, num_str, rate_str,
                    2, start_str, end_str, q1, q2]
                cur_flow_data.append(row)

        self.data['flow'].extend(cur_flow_data)
        res_df = pd.DataFrame(cur_flow_data, columns=['run'] + self.col_map['flow']) # for test
        return res_df

    def generate_cross(self):
        """Generate cross traffic configs.

        Cross config: 'src', 'dst', 'num', 'mode', 'cross_rate_mbps', 'edge_rate_mbps',
                    'hurst', 'mean_duration', 'start', 'end'.
        """
        # TODO: cross traffic rate: cannot be relative to bw, as bw is not determined
        #       now! Ideally, it must be a ratio!
        run_str = self.group['run_str']
        cross_rate_str = 'C(100:100:1001)'
        duration_str = 'N(0.547 0.1)'
        hurst_str = 'U(0.5 0.9)'
        start, end = self.group['sim_start'], self.group['sim_end']
        cur_run_data = []
        for side in range(2):
            assert len(self.group['gw_mid'][side]) > 0
        for src, dst in zip(self.group['gw_mid'][0], self.group['gw_mid'][1]):
            row = [run_str, src, dst, 1, 'ppbp', cross_rate_str, 1000, hurst_str,
                duration_str, start, end]
            cur_run_data.append(row)
        
        self.data['cross'].extend(cur_run_data)
        res_df = pd.DataFrame(cur_run_data, columns=['run'] + self.col_map['cross']) # for test
        return res_df

    def generate(self, btnk_groups=[2, 6, 10, 14, 18, 22],
                 n_run=10, sim_start=0.0, sim_end=600.0, n_leaf=None):
        for typ in self.col_map.keys():
            self.output_csv(typ, is_spec=False)
        for n_btnk in btnk_groups:
            self.init_group(n_btnk, n_run, sim_start, sim_end)
            self.generate_link(n_leaf)
            self.generate_flow()
            self.generate_cross()
        for typ in self.col_map.keys():
            self.output_csv(typ, is_spec=True)

class ConfigGeneratorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.cgen = ConfigGenerator('cgen_test', 'cgen_test')
        self.cgen.init_group(2)
        return super().setUp()
    
    def test_generate_link(self):
        # test: length, src, dst, position <-> bw <-> delay <-> q_monitor
        # leaves number is undetermined, so can only ensure the gw, mid
        # numbers
        link_df = self.cgen.generate_link()
        leaves = self.cgen.group['leaves']
        gws = [set(map(lambda x: self.cgen.group['leaf_gw'][x],
            self.cgen.group['leaves'][i])) for i in range(2)]
        mids = [[], []]
        for x, y in zip(self.cgen.group['gw_mid'][0],
                        self.cgen.group['gw_mid'][1]):
            if x in gws[0]:
                mids[0].append(y)
            elif y in gws[1]:
                mids[1].append(x)
            self.assertNotEqual(x in gws[1] or y in gws[0], True)
        self.assertEqual(len(gws[0]) == len(mids[0]) == len(gws[1])
                        == len(mids[1]) == 2, True)

        small_delay_str, large_delay_str = 'N(0.5 0.1)', 'L(2.9 1.3225)'
        small_bw_str, large_bw_str = 'C(100:100:1001)', '2000'
        qsize_str = 'C(100:100:1001)'
        for i, row in link_df.iterrows():
            self.assertTrue(row.run == self.cgen.group['run_str'] and
                            row.type == 'ppp' and row.q_size == qsize_str and
                            row.q_type == 'C(pie codel)')
            if row.position == 'leaf':
                self.assertTrue(row.dst in gws[0] or row.dst in gws[1])
                self.assertTrue(
                    row.position == 'leaf' and row.q_monitor == 'none' and
                    row.delay_ms == small_delay_str and
                    row.bw_mbps == large_bw_str)
            elif row['src'] in gws[0] or row['dst'] in gws[1]:
                if row['src'] in gws[0]:
                    self.assertIn(row['dst'], mids[0])
                    pos = 'left_mid'
                else:
                    self.assertIn(row['src'], mids[1])
                    pos = 'right_mid'
                self.assertTrue(
                    row.position == pos and row.q_monitor == 'tx' and
                    row.delay_ms == small_delay_str and
                    row.bw_mbps == small_bw_str)
            if row.position == 'mid':
                src = '[' + ' '.join(map(str, mids[0])) + ']'
                dst = '[' + ' '.join(map(str, mids[1])) + ']'
                self.assertTrue(
                    row.src == src and row.dst == dst and
                    row.q_monitor == 'none' and
                    row.delay_ms == large_delay_str and
                    row.bw_mbps == large_bw_str)
        print(link_df)

    def test_generate_flow(self):
        # 2 leaves per gw, 2 btnk, as typical
        self.cgen.generate_link(n_leaf=2)
        flow_df = self.cgen.generate_flow()
        n_src = len(self.cgen.group['leaves'][0])
        n_dst = len(self.cgen.group['leaves'][1])
        self.assertEqual(len(flow_df), n_src * n_dst)
        gw_map = {0: 2, 1:2, 8:10, 9:10, 4:6, 5:6, 12:14, 13:14}
        i = 0
        res = []
        for src in [0, 1, 8, 9]:
            for dst in [4, 5, 12, 13]:
                row = ['[0-9]', src, dst, gw_map[src], gw_map[dst], 'P(50.0 0.44)',
                       'L(3 1.3225)', 2, 'U(0 180)', 'U(420 600)',
                       src // 4, dst // 4]
                self.assertTrue((flow_df.iloc[i] ==
                    pd.Series(row, index=flow_df.columns)).all())
                i += 1
        print(flow_df)

    def test_generate_cross(self):
        self.cgen.generate_link(n_leaf=2)
        self.cgen.generate_flow()
        cross_df = self.cgen.generate_cross()
        for i, (src, dst) in enumerate(zip([2, 7, 10, 15], [3, 6, 11, 14])):
            row = ['[0-9]', src, dst, 1, 'ppbp',
                    'C(100:100:1001)', 1000, 'U(0.5 0.9)', 'N(0.547 0.1)',
                    0, 600]
            self.assertTrue((cross_df.iloc[i] ==
                pd.Series(row, index=cross_df.columns)).all())
        print(cross_df)

    def test_files(self):
        # maybe this topology doesn't work for ns-3 due to 10000 links
        # in the middle and 200 * 200 flows...
        self.cgen = ConfigGenerator('cgen_test', 'inte')
        self.cgen.generate(btnk_groups=[10, 100], n_leaf=2)
        path = '../BBR_test/ns-3.27/edb_configs/cgen_test'
        lengths = {
            'link': 10 * 4 + 1 + 100 * 4 + 1,
            'flow': 400 + 40000,
            'cross':20 + 200
        }
        for typ in ['link', 'flow', 'cross']:
            for csv in [f'inte_{typ}.csv', f'inte_spec_{typ}.csv']:
                df = pd.read_csv(os.path.join(path, csv), index_col=False)
                if 'spec' not in csv:
                    self.assertTrue(df.empty)
                else:
                    self.assertEqual(len(df), lengths[typ])
                    last_run = df.iloc[-1].run
                    self.assertEqual(last_run, f'[10-19]')

def suite():
    suite = unittest.TestSuite()
    suite.addTest(ConfigGeneratorTest('test_generate_link'))
    suite.addTest(ConfigGeneratorTest('test_generate_flow'))
    suite.addTest(ConfigGeneratorTest('test_generate_cross'))
    suite.addTest(ConfigGeneratorTest('test_files'))
    return suite

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Config generator'
        'This tool generates config csvs supported by inflator, specifically '
        'link, flow, cross config (not yet wifi config). Across all config, '
        'the configs are grouped by the number of bottleneck links, and flow/cross '
        'are generated based on the topology in each group. Each group contains several '
        'runs.')
    
    arg_grp = parser.add_mutually_exclusive_group(required=True)
    parser.add_argument('--folder', '-f', type=str, default=None,
                        help='Folder to store configs')
    parser.add_argument('--tag', '-t', type=str, default='config_gen',
                        help='Tag for the config')
    arg_grp.add_argument('--btnk_group', '-b', type=int, nargs='+',
                        help='Number of bottleneck links in each group')
    parser.add_argument('--n_run', '-n', type=int, default=10,
                        help='Number of runs in each group')
    parser.add_argument('--n_leaf', '-l', type=int,
                        help='Number of leaves per gateway')
    parser.add_argument('--start', '-s', type=float, default=0.0,
                        help='Simulation start time (s)')
    parser.add_argument('--end', '-e', type=float, default=600.0,
                        help='Simulation end time (s)')
    arg_grp.add_argument('--test', action='store_true', default=False,
                        help='Run unittests')
    args = parser.parse_args()

    if args.folder is None:
        args.folder = args.tag

    if args.test:
        runner = unittest.TextTestRunner()
        runner.run(suite())
    else:
        cgen = ConfigGenerator(args.folder, args.tag)
        cgen.generate(args.btnk_group, args.n_run, args.start, args.end, args.n_leaf)
