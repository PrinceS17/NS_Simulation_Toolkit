"""
Copyright (c) 2022, IETF Trust and the persons identified as authors of the code.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

from math import ceil
import numpy as np
import pandas as pd
from collections import deque

"""
This is an implementation of the Shared Bottleneck Detection (SBD) algorithm
presented in Online Identification of Groups of Flows Sharing a Network Bottleneck
(IEEE ToN'20).
"""

"""
System design overview

1.  Input: OWD arrays of flows with a specified time unit;
    Output 1: the estimate of the summary statistics per flow, i.e. 
        skew_est, var_est, freq_est;
    Output 2: flow grouping result, format TBD.
"""

class SBDAlgorithm:
    def __init__(self):
        """Initializes with the default parameter values given by RFC 8382, Sec 2.2.
        """
        self.T = 0.35           # interval duration
        self.N = 50             # number of intervals to look over for freq_est
        self.M = 30             # number of intervals for skew_est and var_est
        self.c_s = 0.1
        self.c_h = 0.3
        self.p_s = 0.15         # grouping threshold for skew_est
        self.p_v = 0.7          # threashold used in freq_est
        self.p_f = 0.1          # grouping threshold for freq_est
        self.p_mad = 0.1        # grouping threshold for var_loss
        self.p_d = 0.1          # grouping threshold for pkt_loss
        self.F = 20             # number of sample in the flat portion used in the moving avg

    def _summarize_owd(self, owds, K):
        """Summarizes OWD from data, i.e. averages them over K raw samples.
        Returns the averaged OWD.
        """
        avg_owds, window = [], []
        for i, owd in enumerate(owds):
            window.append(owd)
            if i % K == K - 1 or i == len(owds) - 1:
                avg_owds.append(np.mean(window))
                window.clear()
        # self.d_base = np.mean(window[-self.M:])     # should be used in stream
        return avg_owds
    
    def _piecewise_linear_weighted_moving_average(self, samples, Ks):
        """Given a sample series, return the piecewise linear weighted moving average
        defined in RFC 8382, Sec 4.1.

        TODO: test len(samples) < = > self.F
        """
        if not len(samples):
            return 0
        samples, Ks = np.array(samples), np.array(Ks)
        mf = max(len(samples) - self.F + 1, 1)      # support samples less than F too
        dec_f = np.linspace(mf - 1, 1, mf - 1)
        s = mf * sum(samples[:self.F]) + sum(dec_f * samples[self.F:])
        # n = mf * (len(samples) - mf + 1) + (len(samples) - mf) * (mf - 1)
        n = mf * sum(Ks[:self.F]) + sum(dec_f * Ks[self.F:])    # in case different K in the end
        return s / n

    def _owd_process(self, flow_owds, K):
        """Processes OWDs of a single flow.
        """
        # avg_owds = self._summarize_owd(flow_owds, K)
        window = []
        d_avg_queue = deque(maxlen=self.M)        # for average over M avg_owds
        skew_queue = deque(maxlen=self.M)
        var_queue = deque(maxlen=self.M)
        cross_queue = deque(maxlen=self.N)
        Ks = deque(maxlen=self.M)
        d_avg, x_h = 0, None
        res = {'skew_est': [], 'var_est': [], 'freq_est': []}
        for i, owd in enumerate(flow_owds[::-1]):
            window.append(owd)
            if not (i % K == K - 1 or i == len(flow_owds) - 1):
                continue

            d_avg = np.mean(window)
            d_avg_queue.append(d_avg)
            d_base = np.mean(d_avg_queue)
            Ks.append(len(window))

            # calculates skew_est
            skew_base = sum(map(lambda d: 1 if d < d_base else -1 if d > d_base else 0, window))
            skew_queue.append(skew_base)
            skew_est = self._piecewise_linear_weighted_moving_average(skew_queue, Ks)
            res['skew_est'].append(skew_est)

            # calculates var_est
            var_base = sum(map(lambda d: abs(d - d_avg), window))
            var_queue.append(var_base)
            var_est = self._piecewise_linear_weighted_moving_average(var_queue, Ks)
            res['var_est'].append(var_est)

            # calculates freq_est
            # TODO: freq_est seems need an initial outstanding x_h, as defined below
            var_thd = self.p_v * var_est
            if d_avg < d_base - var_thd and (x_h is None or x_h == 1):
                x, x_h = -1, -1
            elif d_avg > d_base + var_thd and (x_h is None or x_h == -1):
                x, x_h = 1, 1
            else:
                x = 0
            cross_queue.append(x)
            freq_est = np.mean(list(map(abs, cross_queue)))
            res['freq_est'].append(freq_est)
            window.clear()

        return res
    
    def _loss_process(self, flow_drops, K):
        """Processes the losses of given flow. K is the number of samples within 
        an interval T.
        """
        K = int(K)
        res = {'pkt_loss': []}
        loss_queue = deque(maxlen=self.N)
        Ks = deque(maxlen=self.N)
        for i in range(ceil(len(flow_drops) / K)):
            lo, hi = i * K, min((i + 1) * K, len(flow_drops))
            samples = flow_drops[len(flow_drops)-hi:len(flow_drops)-lo]     # first the closest
            loss_queue.append(sum(samples))         # assuming loss is 1
            Ks.append(hi - lo)
            pkt_loss = self._piecewise_linear_weighted_moving_average(loss_queue, Ks)
            res['pkt_loss'].append(pkt_loss)
        return res

    def stream_process(self, flows, t_unit=0.01):
        """Processes the flows inputs and obtains the results.

        Args:
            flows (DataFrame): DataFrame of flows including OWD & loss data.
            t_unit (float): time unit used in the flow data.

        Returns:
            DataFrame: result with skew_est, var_est, freq_est, pkt_loss.
        """
        K = round(self.T / t_unit)
        for col in ['flow', 'owd', 'drop']:
            assert col in flows.columns
        res_df = pd.DataFrame(columns=['flow', 'skew_est', 'var_est', 'freq_est', 'pkt_loss'])
        for flow in flows['flow'].unique():
            owds = flows.query(f'flow == {flow}')['owd']
            drops = flows.query(f'flow == {flow}')['drop']
            res1 = self._owd_process(owds, K)
            res2 = self._loss_process(drops, K)
            for k in res1:
                assert len(res2['pkt_loss']) == len(res1[k]) == ceil(len(owds) / K) > 0
            res1.update(res2)
            flow_df = pd.DataFrame(res1)            
            flow_df['flow'] = flow
            res_df = res_df.append(flow_df)
        return res_df


    def clustering(self, res_df):
        """TODO: clustering technique based on the metric we get.
        """
        pass

