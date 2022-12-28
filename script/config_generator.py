import argparse
import itertools
import numpy as np
import os
import pandas as pd
import sys
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

    def __init__(self, folder, tag, note, root='../BBR_test/ns-3.27/edb_configs'):
        """Maintains the dir and csv for output.
        """
        self.folder = os.path.join(root, folder)
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)
        self.tag = tag
        self.note = note
        self.next_run = 0
        self.col_map = {
            'link': ['src', 'dst', 'position', 'type', 'bw_mbps', 'delay_ms',
                     'q_size', 'q_type', 'q_monitor'],
            'flow': ['src', 'dst', 'src_gw', 'dst_gw', 'num', 'rate_mbps', 'delayed_ack',
                    'start', 'end', 'q_index1', 'q_index2'],
            'cross': ['src', 'dst', 'num', 'mode', 'cross_bw_ratio', 'edge_rate_mbps',
                    'hurst', 'mean_duration', 'start', 'end'],
        }
        self.data = {}
        for typ, cols in self.col_map.items():
            # self.data[typ] = {k: [] for k in ['run'] + cols}
            self.data[typ] = []

    def init_group(self, n_left_btnk, n_right_btnk, n_run=10, sim_start=0, sim_end=600):
        self.group = {
            'n_btnk': [n_left_btnk, n_right_btnk],
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
    
    def output_cmd(self):
        cmd = ' '.join(['python3'] + sys.argv) + '\n'
        cmd_txt = os.path.join(self.folder, f'{self.tag}_cmd.txt')
        with open(cmd_txt, 'w') as f:
            f.write(cmd)
        print(f'Output cmd: {cmd_txt}')
    
    def output_note(self):
        note = os.path.join(self.folder, f'{self.tag}_note.txt')
        with open(note, 'w') as f:
            f.write(self.note)
        print(f'Output note: {note}')

    def generate_link(self, n_leaf=None, link_str_info={}):
        """Generate link config csv based on the number of bottleneck links.
        Link config columns: 'src', 'dst', 'position', 'type', 'bw_mbps',
            'delay_ms', 'q_size', 'q_type', 'q_monitor'.

        Link structure: left / right leaf, left / right btnk, middle links
        Nodes structure: leaf -> gw -> mid left -> mid right -> gw -> leaf

        link_str_info contains the info for bw and delay. Note that the user can
        choose only update bw or delay, but within each type, he/she needs to
        set all the fields clearly.
        """
        max_leaf = [3, 5]           # server side smaller
        run_str = self.group['run_str']
        qtype_str = 'C(pie codel)'
        qsize_str = 'C(100:100:1001)'              # TODO: queue's distribution
        mids = [[], []]     # [left_mids, right_mids]
        cur_link_data = []

        # BW: side x btnk index, delay: side x leaf index(only leaf allowed for delay)
        small_delay_str, large_delay_str = 'N(0.5 0.1)', 'L(2.9 1.3225)'
        small_bw_str, large_bw_str = 'C(100:100:1001)', '2000'
        cur_link_str_info = {
            'bw': [[small_bw_str for _ in range(self.group['n_btnk'][side])]
                    for side in range(2)],
            'delay': [[small_delay_str for _ in range(self.group['n_btnk'][side])]
                    for side in range(2)],
        }
        # if provided '' or None, fill the default value
        # no format check as the user is internal, just let it crash
        for k in link_str_info:
            for side in [0, 1]:
                for i in range(len(link_str_info[k][side])):
                    if link_str_info[k][side][i]:
                        continue
                    link_str_info[k][side][i] = small_bw_str if k == 'bw' \
                        else small_delay_str
        cur_link_str_info.update(link_str_info)

        # set the configs for leaf and left/right btnk
        for side in range(2):       # left / right
            for i in range(self.group['n_btnk'][side]):
                leaf_delay_str = cur_link_str_info['delay'][side][i]
                btnk_bw_str = cur_link_str_info['bw'][side][i]
                
                # leaf -> gw
                n_leaf_to_use = n_leaf
                if n_leaf is None:
                    n_leaf_to_use = np.random.choice(range(1, max_leaf[side]))
                leaf0, leaf1 = self._alloc_nodes(n_leaf_to_use)
                gw, _ = self._alloc_nodes(1)
                row = [run_str, f'[{leaf0}-{leaf1}]', gw, 'leaf', 'ppp',
                    large_bw_str, leaf_delay_str, qsize_str, qtype_str, 'none']
                cur_link_data.append(row)
                self.group['leaves'][side].extend(range(leaf0, leaf1 + 1))
                for leaf in range(leaf0, leaf1 + 1):
                    self.group['leaf_gw'][leaf] = gw

                # gw -> mid
                mid, _ = self._alloc_nodes(1)
                mids[side].append(mid)
                pos = 'left_mid' if side == 0 else 'right_mid'
                link = [gw, mid] if side == 0 else [mid, gw] 
                row = [run_str, link[0], link[1], pos, 'ppp', btnk_bw_str,
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

    def generate_flow(self, dynamic_ratio=0.33, rate_str=None, num_str=None,
                      start_str=None, end_str=None, n_total_users=None):
        """Generate flow configs.

        Flow config: 'src', 'dst', 'src_gw', 'dst_gw', 'num', 'rate_mbps',
                    'delayed_ack', 'start', 'end', 'q_index1', 'q_index2'.
        
        Y() means the only predefined distribution: YouTube bitrate

        Args:
            dynamic_ratio (float, optional): the ratio of dynamic window to total
                                             window size for flow start and end.
        """
        start, end = self.group['sim_start'], self.group['sim_end']
        dynamic_window = (end - start) * dynamic_ratio
        assert start >= 0 and end - dynamic_window >= 0
        run_str = self.group['run_str']
        # rate_str = 'L(2.5 1.3225)'
        rate_str = 'Y()' if not rate_str else rate_str
        if start_str is None:
            start_str = f'U({start} {start + dynamic_window})'
        if end_str is None:
            end_str = f'U({end - dynamic_window} {end})'
        # From the literature, an edge src server has ~ 80 users at most.
        # We divide it by the number of dst to compute the number of users for each src
        # server, and obtain the power law for user number here.
        # (Note that here ns-3 path is assumed to contain 'num' users.)
        # TODO: the understanding of the figure seems wrong
        #       80 users have 1 servers doesn't mean 1 servers only have 80 users!
        #       80 is too less for an edge server, set to 200 for now
        n_total_users = 200 if n_total_users is None else n_total_users
        if num_str is None:
            user_per_dst = n_total_users / len(self.group["leaves"][1])
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

    def generate_cross(self, cross_bw_ratio=None):
        """Generate cross traffic configs.

        Cross config: 'src', 'dst', 'num', 'mode', 'cross_bw_ratio', 'edge_rate_mbps',
                    'hurst', 'mean_duration', 'start', 'end'.
        """
        # TODO: cross traffic rate: cannot be relative to bw, as bw is not determined
        #       now! Ideally, it must be a ratio!
        run_str = self.group['run_str']        
        duration_str = 'N(0.547 0.1)'
        hurst_str = 'U(0.5 0.9)'
        start, end = self.group['sim_start'], self.group['sim_end']
        cross_bw_ratio = 'U(0.05 0.95)' if not cross_bw_ratio else cross_bw_ratio
        cur_run_data = []
        for side in range(2):
            assert len(self.group['gw_mid'][side]) > 0
        for src, dst in zip(self.group['gw_mid'][0], self.group['gw_mid'][1]):
            row = [run_str, src, dst, 1, 'ppbp', cross_bw_ratio, 1000, hurst_str,
                duration_str, start, end]
            cur_run_data.append(row)
        
        self.data['cross'].extend(cur_run_data)
        res_df = pd.DataFrame(cur_run_data, columns=['run'] + self.col_map['cross']) # for test
        return res_df

    def record_output(func):
        def wrapper(self, *args, **kwargs):
            for typ in self.col_map.keys():
                self.output_csv(typ, is_spec=False)
            func(self, *args, **kwargs)
            for typ in self.col_map.keys():
                self.output_csv(typ, is_spec=True)
            self.output_cmd()
            self.output_note()
        return wrapper

    @record_output
    def generate(self, left_btnk_groups, right_btnk_groups, match_btnk=False,
                 n_run=10, sim_start=0.0, sim_end=60.0, n_leaf=None):
        btnk_grp = None
        if match_btnk:
            assert len(left_btnk_groups) == len(right_btnk_groups)
            btnk_grp = zip(left_btnk_groups, right_btnk_groups)
        else:
            btnk_grp = itertools.product(left_btnk_groups, right_btnk_groups)
        for i, (n_left_btnk, n_right_btnk) in enumerate(btnk_grp):
            self.init_group(n_left_btnk, n_right_btnk, n_run, sim_start, sim_end)
            self.generate_link(n_leaf)
            self.generate_flow()
            self.generate_cross()

    @record_output
    def generate_train_w_left_btnk(self, left_btnk_groups, right_btnk_groups,
                                   match_btnk=False, n_run=4,
                                   sim_start=0.0, sim_end=60.0):
        """Generate train set with left btnk. The btnk bw and cross bw ratio are
        set to get the same available bandwidth as (500-1000) Mbps with
        (0.5, 0.8) ratio.
        """
        btnk_grp = None
        if match_btnk:
            assert len(left_btnk_groups) == len(right_btnk_groups)
            btnk_grp = zip(left_btnk_groups, right_btnk_groups)
        else:
            btnk_grp = itertools.product(left_btnk_groups, right_btnk_groups)
        for i, (n_left_btnk, n_right_btnk) in enumerate(btnk_grp):
            link_str_info = {'bw': [['C(250:50:501)'] * n_left_btnk,
                             ['C(1000:100:1501)'] * n_right_btnk]}
            self.init_group(n_left_btnk, n_right_btnk, n_run, sim_start, sim_end)
            self.generate_link(link_str_info=link_str_info)
            self.generate_flow(n_total_users=400)
            self.generate_cross(cross_bw_ratio='U(0.05 0.2)')

    @record_output
    def generate_train_w_right_btnk(self, left_btnk_groups, right_btnk_groups,
                                    match_btnk=False, n_run=4,
                                    sim_start=0.0, sim_end=60.0):
        """Generate train set with right btnk. Note that the # right btnk should
        better not exceed 10, otherwise the left btnk bw would be too small."""
        btnk_grp = None
        if match_btnk:
            assert len(left_btnk_groups) == len(right_btnk_groups)
            btnk_grp = zip(left_btnk_groups, right_btnk_groups)
        else:
            btnk_grp = itertools.product(left_btnk_groups, right_btnk_groups)
        for i, (n_left_btnk, n_right_btnk) in enumerate(btnk_grp):
            link_str_info = {'bw': [['C(2000:100:2101)'] * n_left_btnk,
                             ['C(100:50:200)'] * n_right_btnk]}
            self.init_group(n_left_btnk, n_right_btnk, n_run, sim_start, sim_end)
            self.generate_link(link_str_info=link_str_info)
            self.generate_flow(n_total_users=600)   # increase to ensure right btnk
            self.generate_cross(cross_bw_ratio='U(0.05 0.2)')

    @record_output
    def generate_one_to_n(self, n_run=4, sim_start=0.0, sim_end=60.0):
        """One to N topology for basic test set.
        """
        n_leaf = 2
        left_btnk_groups, right_btnk_groups = [1], [2, 4, 6]
        btnk_grp = itertools.product(left_btnk_groups, right_btnk_groups)
        for i, (n_left_btnk, n_right_btnk) in enumerate(btnk_grp):
            link_str_info = {'bw': [['C(300:100:401)'], [''] * n_right_btnk]}
            self.init_group(n_left_btnk, n_right_btnk, n_run, sim_start, sim_end)
            self.generate_link(n_leaf, link_str_info=link_str_info)
            self.generate_flow()
            self.generate_cross(cross_bw_ratio='U(0.5 0.8)')

    @record_output
    def generate_path_lag_scan(self, n_run=4, sim_start=0.0, sim_end=60.0):
        """Generate path lag test set using 2x2 architecture.
        """
        n_leaf = 1
        left_btnk_groups, right_btnk_groups = [2] * 6, [2] * 6
        btnk_grp = zip(left_btnk_groups, right_btnk_groups)
        for i, (n_left_btnk, n_right_btnk) in enumerate(btnk_grp):
            link_str_info = {
                'bw': [['N(200 5)'] * 2, [''] * 2],
                'delay': [[''] * 2, ['10', f'N({(i + 1) * 20} 3)']]
            }
            self.init_group(n_left_btnk, n_right_btnk, n_run, sim_start, sim_end)
            self.generate_link(n_leaf, link_str_info=link_str_info)
            self.generate_flow(rate_str='C(2.5 5 8)', num_str='N(10 1)')
            self.generate_cross(cross_bw_ratio='U(0.5 0.6)')

    @record_output
    def generate_cross_load_scan(self, n_run=4, sim_start=0.0, sim_end=60.0,
                                 n_leaf=None):
        """Generate cross load test set using 1 to 6 topology.
        """
        left_btnk_groups, right_btnk_groups = [1], [6] * 5
        btnk_grp = itertools.product(left_btnk_groups, right_btnk_groups)
        for i, (n_left_btnk, n_right_btnk) in enumerate(btnk_grp):
            link_str_info = {'bw': [['N(300 5)'], [''] * 6]}
            self.init_group(n_left_btnk, n_right_btnk, n_run, sim_start, sim_end)
            self.generate_link(n_leaf, link_str_info=link_str_info)
            self.generate_flow(rate_str='C(2.5 5 8)', num_str='N(10 1)')
            self.generate_cross(cross_bw_ratio=0.1 + i * 0.2)

    @record_output
    def generate_large_flow_num(self, n_run=4, sim_start=0.0, sim_end=60.0):
        """Generate large number of flow test set.
        Scenario: right btnk, run 1~4 w/ 2 btnks, run 5~8 w/ 6 btnks.
        Within each group, scan the number of flows.
        """
        left_btnk_groups, right_btnk_groups = [2] * 3, [6] * 3
        btnk_grp = zip(left_btnk_groups, right_btnk_groups)
        n_user_choice = [400, 600, 800]
        for i, (n_left_btnk, n_right_btnk) in enumerate(btnk_grp):
            link_str_info = {'bw': [['N(1000 5)'] * n_left_btnk,
                            ['C(100:100:301)'] * n_right_btnk]}
            self.init_group(n_left_btnk, n_right_btnk, n_run, sim_start, sim_end)
            self.generate_link(link_str_info=link_str_info)
            self.generate_flow(n_total_users=n_user_choice[i % 3])
            self.generate_cross(cross_bw_ratio='U(0.3 0.7)')   # then left btnk > 300M

    @record_output
    def generate_para_btnk(self, n_run=4, sim_start=0.0, sim_end=60.0):
        """Generate parallel btnk test set.
        Scenario: scan the number of the right btnks.
        """
        left_btnk_groups, right_btnk_groups = [2] * 4, [4, 8, 12, 16]
        btnk_grp = zip(left_btnk_groups, right_btnk_groups)
        for i, (n_left_btnk, n_right_btnk) in enumerate(btnk_grp):
            link_str_info = {'bw': [['N(2000 5)'] * n_left_btnk,
                             ['C(100:100:201)'] * n_right_btnk]}
            self.init_group(n_left_btnk, n_right_btnk, n_run, sim_start, sim_end)
            self.generate_link(n_leaf=2, link_str_info=link_str_info)
            self.generate_flow(rate_str='C(2.5 5 8)', num_str='25')
            self.generate_cross(cross_bw_ratio='U(0.2 0.5)')


class ConfigGeneratorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.cgen = ConfigGenerator('cgen_test', 'cgen_test')
        self.cgen.init_group(2, 2)
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
        for src in [0, 1, 4, 5]:
            for dst in [8, 9, 12, 13]:
                row = ['[0-9]', src, dst, gw_map[src], gw_map[dst], 'P(50.0 0.44)',
                       'L(2.5 1.3225)', 2, 'U(0 198.0)', 'U(402.0 600)',
                       src // 4, dst // 4]
                self.assertTrue((flow_df.iloc[i] ==
                    pd.Series(row, index=flow_df.columns)).all())
                i += 1
        print(flow_df)

    def test_generate_cross(self):
        self.cgen.generate_link(n_leaf=2)
        self.cgen.generate_flow()
        cross_df = self.cgen.generate_cross()
        for i, (src, dst) in enumerate(zip([2, 6, 11, 15], [3, 7, 10, 14])):
            row = ['[0-9]', src, dst, 1, 'ppbp',
                    'U(0.05 0.95)', 1000, 'U(0.5 0.9)', 'N(0.547 0.1)',
                    0, 600]
            self.assertTrue((cross_df.iloc[i] ==
                pd.Series(row, index=cross_df.columns)).all())
        print(cross_df)

    def test_files(self):
        # maybe this topology doesn't work for ns-3 due to 10000 links
        # in the middle and 200 * 200 flows...
        self.cgen = ConfigGenerator('cgen_test', 'inte')
        self.cgen.generate(left_btnk_groups=[10], right_btnk_groups=[10, 100],
                           n_leaf=2)
        path = '../BBR_test/ns-3.27/edb_configs/cgen_test'
        lengths = {
            'link': 10 * 4 + 1 + 100 * 2 + 20 + 1,
            'flow': 400 + 4000,
            'cross':20 + 110
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
    arg_grp.add_argument('--tag', '-t', type=str, default='config_gen',
                        help='Tag for the config')
    parser.add_argument('--profile', '-p', type=str, default='',
                        choices=['', 'left-btnk', 'right-btnk', 'one-to-n',
                        'path-lag', 'load-scan', 'large-flow', 'para-btnk'],
                        help='Profile for the config generation')
    parser.add_argument('--note', '-nt', type=str, default='',
                        help='Note for the current config')
    parser.add_argument('--left_btnk_group', '-lb', type=int, nargs='+',
                        help='Number of left bottleneck links in each group')
    parser.add_argument('--right_btnk_group', '-rb', type=int, nargs='+',
                        help='Number of right bottleneck links in each group')
    parser.add_argument('--match_btnk', '-m', action='store_true', default=False,
                        help='Match the number of bottlenecks in left/right group')
    parser.add_argument('--n_run', '-n', type=int, default=10,
                        help='Number of runs in each group')
    parser.add_argument('--n_leaf', '-l', type=int,
                        help='Number of leaves per gateway')
    parser.add_argument('--start', '-s', type=float, default=0.0,
                        help='Simulation start time (s)')
    parser.add_argument('--end', '-e', type=float, default=60.0,
                        help='Simulation end time (s)')
    arg_grp.add_argument('--test', action='store_true', default=False,
                        help='Run unittests')
    args = parser.parse_args()

    if args.folder is None:
        args.folder = args.tag

    if args.test:
        runner = unittest.TextTestRunner()
        runner.run(suite())
        exit(0)

    cgen = ConfigGenerator(args.folder, args.tag, args.note)
    if not args.profile:    
        cgen.generate(args.left_btnk_group, args.right_btnk_group, args.match_btnk,
                      args.n_run, args.start, args.end, args.n_leaf)
    elif args.profile == 'left-btnk':
        cgen.generate_train_w_left_btnk(args.left_btnk_group, args.right_btnk_group,
                      args.match_btnk, args.n_run, args.start, args.end)
    elif args.profile == 'right-btnk':
        cgen.generate_train_w_right_btnk(args.left_btnk_group, args.right_btnk_group,
                      args.match_btnk, args.n_run, args.start, args.end)
    elif args.profile == 'one-to-n':
        cgen.generate_one_to_n(args.n_run, args.start, args.end)
    elif args.profile == 'path-lag':
        cgen.generate_path_lag_scan(args.n_run, args.start, args.end)
    elif args.profile == 'load-scan':
        cgen.generate_cross_load_scan(args.n_run, args.start, args.end, args.n_leaf)
    elif args.profile == 'large-flow':
        cgen.generate_large_flow_num(args.n_run, args.start, args.end)
    elif args.profile == 'para-btnk':
        cgen.generate_para_btnk(args.n_run, args.start, args.end)
