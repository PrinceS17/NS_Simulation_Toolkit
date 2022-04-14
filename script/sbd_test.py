from math import ceil
import unittest

import numpy as np
import pandas as pd
from sbd import SBDAlgorithm

# TODO: the sampling in my simulation is not the same as SBD algorithm
#       in SBD, raw packets are collected within T; in simulation,
#       a fixed interval is used to export different data to csv, and
#       their freshness may not match the interval
#       ? critical or not?

class SbdTest(unittest.TestCase):
    def setUp(self):
        self.sbd = SBDAlgorithm()

    def test_moving_average(self):
        sequences = [
            [], [1], [-2,2],
            np.linspace(1, 9, 9),      # len < F = 20
            np.linspace(-10, 10, 21),   # len > F
            np.linspace(30, 31, 35),    # len ~ K: actually not different
            np.random.random(1000),     # large
        ]
        K = 10
        Ks = [
            [], [K], [K, K],
            [K] * 9,
            [K] * 21,
            [K] * 35,
            [K] * 1000,
        ]
        results = [
            0, 1, 0,
            5,
            -0.2439,
            30.5,
            0.5,
        ]
        results = [r / K for r in results]
        for i, seq in enumerate(sequences):
            val = self.sbd._piecewise_linear_weighted_moving_average(seq, Ks[i])
            if i < 4:
                self.assertEqual(val, results[i], msg=f'Error i = {i}')
            else:
                self.assertAlmostEqual(val, results[i], delta=0.5)
    
    def test_owd_process(self):
        K = 36
        bi, tri = [-1, 1], [0, 2, 4]
        sequences = [
            [], [1], [2,2],                 # short
            np.zeros(K), np.zeros(1000),   # flat
            [1] * K + [-1] * K,           # skew & freq
            bi * K, bi * 40,               # periodic: freq & var
            tri * K, tri * 100,
            np.random.normal(0, 1, 10*K),    # var
            np.random.normal(5, 100, 10000) / 100,
            np.concatenate([np.random.normal(5, 0.5, K), np.random.normal(-5, 0.5, K)]),
            np.sin(0.1 * np.linspace(1, 4 * K, 4 * K)),
        ]
        results = [
            [[]] * 3, [[0]] * 3, [[0]] * 3,
            [[0]] * 3, [[0] * ceil(1000 / K)] * 3,
            [[0, -0.5], [0, 0], [0, 0.5]],
            [[0, 0], [1, 1], [0, 0]], [[0, 0, 0], [1, 1, 1], [0, 0, 0]],
            [[0] * 3, [4/3] * 3, [0] * 3], [[0] * 9, [4/3] * 9, [0] * 9],
            # not precise below
            [[0] * 10, [1] * 10, [0] * 10],
            [[0] * 278, [1] * 278, [0] * 278],
            [[0, 0.5], [0.5, 0.5], [0, -0.5]],
            [[0.5, 0] * 2, [0.5] * 4, [0, 1/2, 2/3, 3/4]]
        ]

        # TODO: dec is 0 for normal distribution
        for i, seq in enumerate(sequences):
            res = self.sbd._owd_process(seq, K)
            res_list = [res['skew_est'], res['var_est'], res['freq_est']]
            dec = 0 if i > 4 else 2
            np.testing.assert_almost_equal(res_list, results[i], decimal=dec)
    
    def test_loss_process(self):
        K = 10
        sequences = [
            [], [1], [0,1],
            [0] * 9 + [1],
            [0] * 5 + [1] * 5 + [0] * 9 + [1]
        ]
        results = [
            [], [1], [0.5],
            [0.1],
            [0.1, 0.3],
        ]
        for i, seq in enumerate(sequences):
            res = self.sbd._loss_process(seq, K)
            res_list = res['pkt_loss']
            np.testing.assert_almost_equal(res_list, results[i], decimal=2)
    
    def test_stream_process(self):
        # TODO: only very simple case here, relying on simulated trace
        flows = pd.DataFrame(columns=['flow', 'owd', 'drop'],
            data=[[0, 0.1, 0]] * 30 + [[0, 0.1, 1]] * 5)
        expected = pd.DataFrame(columns=['flow', 'skew_est', 'var_est', 'freq_est', 'pkt_loss'],
            data=[[0, 0, 0, 0, 1/7]])
        actual = self.sbd.stream_process(flows)
        # np.testing.assert_almost_equal(actual, expected, decimal=2)
        self.assertTrue((actual == expected).all()[0])

def suite():
    suite = unittest.TestSuite()
    suite.addTest(SbdTest('test_moving_average'))
    suite.addTest(SbdTest('test_loss_process'))
    suite.addTest(SbdTest('test_owd_process'))
    suite.addTest(SbdTest('test_stream_process'))
    return suite

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite())