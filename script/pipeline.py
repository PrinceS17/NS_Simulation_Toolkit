import os
from matplotlib import pyplot as plt
import subprocess as sp
import pandas as pd
from sklearn.metrics import adjusted_mutual_info_score
import seaborn as sns

def tick(row, interval=2):
    row1 = row.copy()
    row1['time'] = row1['time'] // interval * interval
    return row1


class Pipeline:
    """The data processing pipeline for the raw time series data, including
    series getter, time averaging, different processing techniques.
    """

    def __init__(self, csv, tag, show=True, folder=None):
        self.df = pd.read_csv(csv, index_col=False)
        self.score_func = {
            'linear': lambda s1, s2: s1.corr(s2),   # i.e. 'pearson' (default)
            'rank': lambda s1, s2: s1.corr(s2, method='spearman'),  # other option: 'kendall'
            'mut_info': lambda s1, s2: adjusted_mutual_info_score(s1, s2),
        }
        self.tag = tag
        self.show = show
        self.folder = folder

    def get_flow_series(self, df, i, signal):
        return df[df.flow == i].reset_index(drop=True).copy()[signal]
    
    def _time_average(self, df, interval, t_unit=0.01):
        """Averages the data over a given interval in second, with the time
        unit in the csv specified.
        """
        df1 = df.copy()
        if not interval:
            return df1
        window = int(interval / t_unit)
        # TODO: warning: sample across some discrete signal & can miss some!
        for signal in ['rtt', 'cwnd', 'bytes_in_flight', 'llr']:
            df1[signal] = df1[signal].transform(lambda x: x.rolling(window=window).mean())
        df1 = df1.dropna().transform(tick, axis=1, interval=interval)
        df1['interval'] = [interval] * df1.shape[0]
        return df1.drop_duplicates(subset=['flow', 'time'])

    def calculate_scores(self, i, j, intervals=[None]):
        """Calculates scores given flow index i and j for all score func & signal.
        """
        res = pd.DataFrame(columns=['signal', 'corr', 'interval', 'score'])
        for corr, f_score in self.score_func.items():
            for interval in intervals:
                for signal in self.df.columns:
                    if signal in ['time', 'flow']:
                        continue
                    df = self._time_average(self.df, interval)
                    s1 = self.get_flow_series(df, i, signal)
                    s2 = self.get_flow_series(df, j, signal)
                    if abs(len(s1) - len(s2)) < 2:
                        if len(s1) < len(s2):
                            s2 = s2[:len(s1)]
                        else:
                            s1 = s1[:len(s2)]
                    assert len(s1) == len(s2), \
                        f's1, s2 have quite different length: {interval}, {signal}, {len(df)}, {len(s1)} != {len(s2)}'
                    score = f_score(s1, s2)
                    row = pd.DataFrame([[signal, corr, interval, score]],
                        columns=res.columns)
                    res = res.append(row)
        print(res.to_string())
        return res 

    def show_or_save(self, fname):
        if self.show:
            plt.show()
        else:
            folder = self.folder if self.folder else '.'
            path = os.path.join(folder, fname)
            plt.savefig(path)
            print(f'Figure saved: {path}')
        plt.clf()

    def plot_score_vs_signal_w_func(self, i, j):
        """Reduntantly keep the function for customization.
        """
        res = self.calculate_scores(i, j)
        plt.close()
        sns.barplot(x='signal', y='score', hue='corr', data=res)
        self.show_or_save(f'score_vs_signal_w_f_{self.tag}.png')
    
    def plot_score_vs_intervals_w_signal(self, i, j, intervals, corr):
        res = self.calculate_scores(i, j, intervals=intervals)
        res = res[res['corr'] == corr]
        plt.close()
        ax = sns.barplot(x='interval', y='score', hue='signal', data=res)
        ax.set_title(f'Score vs intervals w/ signal, corr = {corr}')
        self.show_or_save(f'score_vs_interval_w_signal_{self.tag}.png')
    
    def plot_flows(self, show_flow=None):
        df = self.df[self.df.flow < show_flow] if show_flow else self.df
        for field in df.columns:
            if field in ['time', 'flow']:
                continue
            plt.close()
            sns.lineplot(x='time', y=field, hue='flow', data=df)
            self.show_or_save(f'flow_{field}_{self.tag}.pdf')



for tag in ['33', '34']:
    ppl =Pipeline(f'all-data_{tag}.csv', f'test-{tag}', show=False)
    # ppl.plot_score_vs_signal_w_func(0, 1)
    ppl.plot_flows(2)



# for func in ['linear', 'rank', 'mut_info']:
#     ppl.plot_score_vs_intervals_w_signal(0, 1, [0.02, 0.1], func)