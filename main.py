# Optimal Estimation - HW3 - Battery Equivalent Circuit Model Design

import pandas as pd
import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt

def read_data(file_name: str):
    '''
    Temp
    '''
    data = pd.read_csv(file_name)
    data['time (h:m:s)'] = pd.to_timedelta(data['time (h:m:s)']).dt.total_seconds()
    time = data['time (h:m:s)'].to_numpy()
    voltage = data['voltage (volts)'].to_numpy()
    current = data['current (amps)'].to_numpy()
    t0 = time[0]
    time = time - t0
    t0_new = time[0]
    time = [t + 86400 if t <= 0 else t for t in time]
    time[0] = t0_new
    return time, voltage, current

def find_periods(current: np.array):
    '''
    Temp
    '''
    period_cutoffs = []
    periods = []
    curr_old = 0
    for i, curr in enumerate(current):
        if curr - curr_old > 10:
            period_cutoffs.append([i, curr])
        curr_old = curr
    for i, cutoff in enumerate(period_cutoffs):
        if i == 0:
            cutoff_old = period_cutoffs[i][0]
        else:
            periods.append([cutoff_old, cutoff[0] - 1])
            cutoff_old = period_cutoffs[i][0]

    return periods

def estimate_parameters(period: list, current: np.array, voltage: np.array, time: np.array):
    '''
    Temp
    '''
    params = []
    ps = period[0]
    pe = period[1]
    # Find R0
    curr_old = current[int(ps)]
    for i, curr in enumerate(current[ps:pe]):
        if curr < 1:
            idis = curr_old
            ts = i - 1 + ps
            tr = i + ps
            break
        curr_old = curr
    print(f'''idis: {idis}, ts: {ts}, tr: {tr}''')
    print(f'''voltage at tr = 0: {voltage[tr]} and voltage at ts: {voltage[ts]}''')
    R0 = (voltage[tr] - voltage[ts]) / idis
    print(f'''R0: {R0}''')

    # plot_side_by_side(period, current, voltage, ts, tr)
    x0_1 = np.array([25, 1, 10])
    # print(np.array(time[tr:pe]))
    time_zero_period = (np.array(time[tr:pe]) - tr * 10)
    result_1 = sp.optimize.least_squares(fun_1, x0_1, args=(idis, time_zero_period, voltage[tr:pe]))
    print(result_1.x)
    return params

def soc_stuff(current):
    c0 = current[0]
    curr_old = c0
    SOC = []
    dt = 10
    for curr in current:
        if curr == curr_old:
            pass
        SOC.append(dt * 1 / 2 * (curr + curr_old))
        curr_old = curr
    Q = sum(SOC)
    plt.plot(current)
    plt.show()
    return 1

def plot_side_by_side(period, current, voltage, ts, tr):
    '''
    Plots
    '''
    ps = period[0]
    pe = period[1]
    fig, ax1 = plt.subplots()
    color = 'tab:red'
    ax1.set_ylabel('voltage', color=color)
    ax1.plot(np.arange(ps, pe, 1), voltage[ps:pe], color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel('current', color=color)
    ax2.plot(np.arange(ps, pe, 1), current[ps:pe], color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.vlines(ts, 0, 70, colors='orange')
    plt.vlines(tr, 0, 70, colors='green')
    plt.show()
    return 1

def fun_1(x, i_ts, tr, v_tr):
    return x[0]- i_ts * x[1] * np.exp(-tr / (x[1] * x[2])) - v_tr

# def fun_1(OCV, R1, C1, tr, i_ts, v_ts):
#     return OCV - i_ts * np.exp(-tr / (R1 * C1)) - v_ts

def main():

    file_name = 'pulse_discharge_test_data.csv'
    time, voltage, current = read_data(file_name)
    print(time[0:100])
    periods = find_periods(current)
    for period in periods:
        params = estimate_parameters(period, current, voltage, time)
        # temp break to only test first rest period
        break

    return 1

if __name__ == '__main__':
    main()