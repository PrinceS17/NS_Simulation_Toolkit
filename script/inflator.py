import numpy as np
import pandas as pd

def choice_of_ytb_bitrate():
    """Draws a bitrate from Youtube's bitrate distribution."""
    bitrate = [0.08, 0.4, 1, 2.5, 5, 8, 10, 40]
    p = np.array([4.9, 13.3, 23.7, 33.1, 18.15, 6.75, 0.05, 0.05]) / 100
    return np.random.choice(bitrate, 1, p=p)

def isfloat(val):
    """Check if a string is a float."""
    try:
        float(val)
        return True
    except ValueError:
        return False

def inflate_rows(df_config, is_test):
    """
    Inflate aggregated fields, i.e. broadcast the element in the 
    aggregated fields to multiple rows. Supported grammars include:
        1. Run inflation, e.g. [2], [1 4] or [1-4] in the run field.
            and * for automated generation.
        2. Cartesian field inflation, e.g. [1-3], [1:1:4] in for any fields,
            and the effect is inflate the Cartesian product of all the
            aggregated fields.
        3. Static field inflation, e.g. {1:1:4}, and the effect is for
            any static fields, assert the number of rows is the same,
            and inflate it only once. For instance, {1:1:4} in src and
            {2:1:5} in dst will generate 4 rows in total but not 16.
        4. Automated run generation: not well defined now. Currently
            allows it only when static fields are used, and one static
            field group will be regarded as a single run only when it
            includes src/dst field, as otherwise there's no way to scan runs
            for several links.
        5. Random field: support three types of distributions:
            N(mean std): positive samples drawn from N(mean, std);
            U(start end): unifrom distribution;
            C(start end step): choice drawn from [start:end:step].
    """

    def _draw_sample(col, val):
        assert val[0] in ['N', 'U', 'C', 'L', 'P', 'Y'], \
                'Distribution not supported.'
        tmp = ''
        if ' ' in val[1:] and val[0] != 'C':
            tmp = eval(val[1:].replace(' ', ','))
        if val[0] == 'N':
            assert tmp[0] > 0, 'Mean must be positive.'
            vals = np.random.normal(tmp[0], tmp[1], 1)
            while vals[0] <= 0:
                vals = np.random.normal(tmp[0], tmp[1], 1)
        elif val[0] == 'U':
            assert tmp[0] >= 0
            vals = np.random.uniform(tmp[0], tmp[1], 1)
        elif val[0] == 'C':
            tmp = val[2:-1].split(' ')
            if ' ' in val:      # use ' ' grammar only for enumeration
                candidates = tmp
                if isfloat(tmp[0]):
                    candidates = list(map(float, candidates))
            elif ':' in val[1:]:
                tmp = eval(val[1:].replace(':', ','))
                candidates = list(range(tmp[0], tmp[2], tmp[1]))
            # else:
            #     # deprecated
            #     tmp = eval(val[1:].replace(' ', ','))
            #     candidates = list(range(tmp[0], tmp[1], tmp[2]))
            vals = np.random.choice(candidates, 1)
        elif val[0] == 'L':
            vals = np.random.lognormal(tmp[0], tmp[1], 1)
            while vals[0] <= 0:
                vals = np.random.lognormal(tmp[0], tmp[1], 1)
        elif val[0] == 'P':    # Power law, P(scale, a i.e. index)
            vals = tmp[0] * np.random.power(tmp[1], 1)
            vals = np.maximum(vals, 1)
        elif val[0] == 'Y':
            vals = choice_of_ytb_bitrate()

        # field type processing
        if col in ['arrival_rate', 'mean_duration', 'pareto_index',
            'hurst', 'delay_ms', 'cross_bw_ratio', 'rate_mbps']:
            # avoid 0 for duration/index/cross_bw_ratio
            return max(round(vals[0], 3), 0.001)
        elif type(vals[0]) in [np.str_, str]:
            return vals[0]
        else:
            new_val = int(np.ceil(vals[0]))
            assert new_val > 0, f'Invalid value: {new_val}'
            return new_val

    def _dfs(cur_row, i, last_run, j_static, increase_run, result):
        # a recursion for row compilation
        # increase run: use by previous field to indicate if we want to
        #              increase the run number in the end
        if i == len(cur_row):
            cur_row_copy = cur_row.copy()
            for j, val in enumerate(cur_row_copy):
                col = df_config.columns[j]
                if type(val) != str or '(' not in val:
                    continue
                cur_row_copy[j] = _draw_sample(col, val)
            if cur_row.run != '*':
                # assert not increase_run
                last_run = int(cur_row.run)
            else:
                cur_row_copy['run'] = last_run + 1
                last_run = last_run + 1 if increase_run else last_run
            result.append(cur_row_copy)
            return last_run

        col, val = df_config.columns[i], cur_row[i]

        # this breaks the rule that only str field gets inflated, so process first
        if col == 'num':
            # for simplicity, don't support nested inflation in 'num' field
            # the sole use case should be random distribution here
            if type(val) == str and not val.isdigit():
                assert not ('[' in val or '{' in val)
                assert '(' in val
                num = _draw_sample(col, val)
            else:
                num = int(val)
            for _ in range(num):
                cur_row[i] = 1
                last_run = _dfs(cur_row, i + 1, last_run, j_static, increase_run,
                                result)
            cur_row[i] = val
            return last_run

        if type(val) != str or ('[' not in val and '{' not in val):
            return _dfs(cur_row, i + 1, last_run, j_static, increase_run, result)

        # inflate the field to multiple values

        assert val[0] in ['[', '{'] and val[-1] in [']', '}'], f'Invalid: {val}'
        # Current compilation itself is not satisfactory, and the most painful
        # part is the (src,dst) must be specified by hand. We are then combining
        # it with the config_generator to do the trick.
        if ' ' in val:
            tmp_vals = val[1:-1].split(' ')
            vals = list(map(int, tmp_vals)) if tmp_vals[0].isdigit() else tmp_vals
        elif '-' in val:
            tmp = eval(val.replace('-', ',').replace('{', '[').replace('}', ']'))
            vals = list(range(tmp[0], tmp[1] + 1))
        elif ':' in val:
            tmp = eval(val.replace(':', ',').replace('{', '[').replace('}', ']'))
            vals = list(range(tmp[0], tmp[2], tmp[1]))
        elif '[' in val or '{' in val:
            tmp_vals = val[1:-1]
            vals = [int(tmp_vals)] if tmp_vals.isdigit() else [tmp_vals]

        # now process Cartesian or static field inflation
        if '{' in val:
            if j_static is None:        # the first static field
                """
                If the 1st static field, then we need to scan it, and set the 
                autogenerated run to the j_static. j_static is used to bind
                the later static field's value.
                """
                cur_row_copy = cur_row.copy()
                res_last_run = -1
                for j_static, v in enumerate(vals):
                    # If it's in src or dst, we assume the run doesn't change.
                    # We need to keep run the same for the inflated row
                    # with the same Cartesian product index. If it's neither
                    # src nor dst, then run should increase.
                    cur_row[i] = vals[j_static]
                    if col not in ['src', 'dst']:
                        increase_run = True
                        last_run = _dfs(cur_row, i + 1, last_run, j_static,
                                        increase_run, result)
                        res_last_run = last_run
                    else:
                        res_last_run = _dfs(cur_row, i + 1, last_run, j_static,
                                            increase_run, result)
                        # The pure static field case: increase run at the end of all inflation,
                        # otherwise last_run is not updated as it's fixed by us.
                        if j_static == len(vals) - 1 and res_last_run == last_run:
                            res_last_run += 1
                cur_row = cur_row_copy
                return res_last_run
            else:                       # the later static field
                cur_row[i] = vals[j_static]
                last_run = _dfs(cur_row, i + 1, last_run, j_static, increase_run,
                                result)
                cur_row[i] = val
                return last_run
        elif '[' in val:
            for v in vals:
                # each new value in Cartesian field should add last_run
                cur_row[i] = v
                if i > 0:
                    increase_run = True
                last_run = _dfs(cur_row, i + 1, last_run, j_static, increase_run,
                                result)
            cur_row[i] = val
            return last_run


    # main loop for row compilation
    is_base = False
    if 'run' not in df_config.columns:
        df_config['run'] = '-1'
        is_base = True

    result_rows, row_idx = [], []
    last_run, j_static, increase_run = -1, None, False
    for i_row, row in df_config.iterrows():
        last_run = _dfs(row, 0, last_run, j_static, increase_run, result_rows)
    result_df = pd.DataFrame(result_rows, columns=df_config.columns)
    print(result_df)
    result_df['run'] = result_df.apply(lambda x: int(x['run']), axis=1)
    result_df = result_df.sort_values(['run', 'src', 'dst']).reset_index(drop=True)

    if is_base:
        result_df.drop(columns=['run'], inplace=True)

        keys = ['src', 'dst']
        for k in ['start', 'end']:
            if k in result_df.columns:
                keys.append(k)
        df1 = result_df.drop_duplicates(subset=keys)
        assert len(df1) == len(result_df), f'Repeated base setting on {keys}'

    if is_test:
        print(result_df)

    return result_df
