# Optimal Estimation - HW3 - Battery Equivalent Circuit Model Design

import pandas as pd
import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt

from collections import OrderedDict

# Matplotlib global variables
plt.rcParams['axes.grid'] = True
mpl.rcParams['legend.loc'] = 'lower right'

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
    ocv_estimate = voltage[pe]
    flag = 0
    for i, (curr, volt) in enumerate(zip(current[ps:pe], voltage[ps:pe])):
        if curr < 1 and flag == 0:
            idis = current[i - 1]
            ts = i - 1 + ps
            tr = i + ps
            flag = 1
        if flag == 1 and volt >= 0.95 * (ocv_estimate - voltage[tr]) + voltage[tr]:
            t95 = i + ps
            break
    # print(f'''idis: {idis}, ts: {ts}, tr: {tr}''')
    # print(f'''voltage at tr = 0: {voltage[tr]} and voltage at ts: {voltage[ts]}''')
    R0 = (voltage[tr] - voltage[ts]) / idis
    # print(f'''R0: {R0}''')
    # Optional plotting for understanding the problem
    # plot_side_by_side(period, current, voltage, ts, tr, t95)
    x0_1 = np.array([ocv_estimate, 10, 10])
    x0_2 = np.array([ocv_estimate, 10, 10, 1, 1])
    x0_3 = np.array([ocv_estimate, 10, 10, 10, 10, 1, 1])
    time_zero_period = (np.array(time[tr:t95]) - tr * 10)
    result_1 = sp.optimize.least_squares(fun_1, x0_1, args=(idis, \
                                                            time_zero_period, \
                                                            voltage[tr:t95]))
    result_2 = sp.optimize.least_squares(fun_2, x0_2, args=(idis, \
                                                            time_zero_period, \
                                                            voltage[tr:t95]))
    result_3 = sp.optimize.least_squares(fun_3, x0_3, args=(idis, \
                                                            time_zero_period, \
                                                            voltage[tr:t95]))
    # print(np.average(result_1.fun))
    # print(np.average(result_2.fun))
    # print(np.average(result_3.fun))
    # print()
    params = (R0, result_1.x, result_2.x, result_3.x)
    residuals = (result_1.fun, result_2.fun, result_3.fun)
    return params, residuals

def soc_calc(current):
    curr_old = current[0]
    Q_int = []
    SOC = [1]
    SOC_old = SOC[0]
    dt = 10
    SOC_total = 0
    for curr in current:
        if curr == curr_old:
            pass
        Q_int.append(dt * 1 / 2 * (curr + curr_old))
        curr_old = curr
    Q = sum(SOC)
    Q = sum(Q_int)
    for val in Q_int:
        SOC_total += val
        SOC.append(SOC_old - SOC_total / Q)
    return np.array(SOC)

def plot_side_by_side(period, current, voltage, ts, tr, t95):
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
    plt.vlines(t95, 0, 70, colors='blue')
    plt.show()
    return 1

def fun_1(x, i_ts, tr, v_tr):
    '''
    x is composed of OCV, R1, C1 in that order
    '''
    return x[0] - i_ts * x[1] * np.exp(-tr / (x[1] * x[2])) - v_tr

def fun_2(x, i_ts, tr, v_tr):
    '''
    x is composed of OCV, R1, C1, R2, C2 in that order
    '''
    return x[0] - i_ts * x[1] * np.exp(-tr / (x[1] * x[2])) \
                - i_ts * x[3] * np.exp(-tr / (x[3] * x[4])) - v_tr

def fun_3(x, i_ts, tr, v_tr):
    '''
    x is composed of OCV, R1, C1, R2, C2, R3, C3 in that order
    '''
    return x[0] - i_ts * x[1] * np.exp(-tr / (x[1] * x[2])) \
                - i_ts * x[3] * np.exp(-tr / (x[3] * x[4])) \
                - i_ts * x[5] * np.exp(-tr / (x[5] * x[6])) - v_tr

def all_the_plotting(time, soc, periods, params_all_1, params_all_2, params_all_3):
    '''
    Temp
    '''
    # RC elements vs SOC plotting
    idx = 0
    fig, ax = plt.subplots(2, 1)
    ax[0].set_title(r'Reistance and Capacitance vs SOC')
    ax[0].set_ylabel(r'Resistance, unit')
    ax[1].set_ylabel(r'Capacitance, unit')
    fig.supxlabel(r'SOC')
    for i, t in enumerate(time):
        if int(t) in [10 * period[0] for period in periods]:
            ax[0].scatter(soc[i], params_all_1[idx, 1], color='g', marker='.', label='1 element, R1')
            ax[0].scatter(soc[i], params_all_2[idx, 1], color='b', marker='.', label='2 elements, R1')
            ax[0].scatter(soc[i], params_all_2[idx, 3], color='r', marker='.', label='2 elements, R2')
            if params_all_3[idx, 0] < 50:
                ax[0].scatter(soc[i], params_all_3[idx, 1], color='c', marker='.', label='3 elements, R1')
                ax[0].scatter(soc[i], params_all_3[idx, 3], color='m', marker='.', label='3 elements, R2')
                ax[0].scatter(soc[i], params_all_3[idx, 5], color='y', marker='.', label='3 elements, R3')
            
            ax[1].scatter(soc[i], params_all_1[idx, 2], color='g', marker='.', label='1 element, R1')
            ax[1].scatter(soc[i], params_all_2[idx, 2], color='b', marker='.', label='2 elements, R1')
            ax[1].scatter(soc[i], params_all_2[idx, 4], color='r', marker='.', label='2 elements, R2')
            if params_all_3[idx, 0] < 50:
                ax[1].scatter(soc[i], params_all_3[idx, 2], color='c', marker='.', label='3 elements, R1')
                ax[1].scatter(soc[i], params_all_3[idx, 4], color='m', marker='.', label='3 elements, R2')
                ax[1].scatter(soc[i], params_all_3[idx, 6], color='y', marker='.', label='3 elements, R3')
            idx += 1
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    ax[0].legend(by_label.values(), by_label.keys())
    ax[1].legend(by_label.values(), by_label.keys())
    plt.show()
    
    # OCV vs SOC plotting
    idx = 0
    fig2 = plt.figure(1)
    plt.suptitle(r'OCV vs SOC')
    plt.xlabel(r'SOC')
    plt.ylabel(r'OCV')
    for i, t in enumerate(time):
        if int(t) in [10 * period[0] for period in periods]:
            plt.scatter(soc[i], params_all_1[idx, 0], color='green', marker='.', label='1 element')
            plt.scatter(soc[i], params_all_2[idx, 0], color='blue', marker='.', label='2 elements')
            if params_all_3[idx, 0] < 50:
                plt.scatter(soc[i], params_all_3[idx, 0], color='red', marker='.', label='3 elements')
            idx += 1
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.show()
    return 1

def main():
    '''
    Temp
    '''
    file_name = 'pulse_discharge_test_data.csv'
    time, voltage, current = read_data(file_name)
    periods = find_periods(current)
    soc = soc_calc(current)
    R_0 = []
    params_all_1 = []
    params_all_2 = []
    params_all_3 = []
    idx = 0
    for period in periods:
        params, residuals = estimate_parameters(period, current, voltage, time)
        if idx == 0:
            R_0 = params[0]
            params_all_1 = params[1]
            params_all_2 = params[2]
            params_all_3 = params[3]
            idx += 1
        else:
            R_0 = np.vstack((R_0, params[0]))
            params_all_1 = np.vstack((params_all_1, params[1]))
            params_all_2 = np.vstack((params_all_2, params[2]))
            params_all_3 = np.vstack((params_all_3, params[3]))
    # print(R_0)
    # print(params_all_1)
    # print(params_all_2)
    # print(params_all_3)
    all_the_plotting(time, soc, periods, params_all_1, params_all_2, params_all_3)
    return 1

if __name__ == '__main__':
    main()