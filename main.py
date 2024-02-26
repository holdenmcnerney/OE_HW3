# Optimal Estimation - HW3 - Battery Equivalent Circuit Model Design

from datetime import datetime
import pandas as pd
import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt


def main():

    data = pd.read_csv('pulse_discharge_test_data.csv')
    data['time (h:m:s)'] = pd.to_timedelta(data['time (h:m:s)']).dt.total_seconds()
    time = data['time (h:m:s)'].to_numpy()
    voltage = data['voltage (volts)'].to_numpy()
    current = data['current (amps)'].to_numpy()

    t0 = time[0]
    time = time - t0
    t0_new = time[0]
    time = [t + 86400 if t <= 0 else t for t in time]
    time[0] = t0_new
    c0 = current[0]
    curr_old = c0
    SOC = []
    dt = 10

    t_old = t0_new
    for t in time:
        if abs(t - t_old) != 10:
            print(f'''{t} {t_old} {abs(t - t_old)}''')
        t_old = t

    # for curr in current:
    #     if curr == curr_old:
    #         pass
    #     SOC.append(dt * 1 / 2 * (curr + curr_old))
    #     curr_old = curr

    # Q = sum(SOC)

    # plt.plot(current)
    # plt.show()

    # sp.optimize.least_squares()

    return 1

if __name__ == '__main__':
    main()