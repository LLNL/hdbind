import numpy as np
import pandas as pd
import os
from pathlib import Path

host="lassen595"
length_list = [1000, 10000, 100000, 1000000]
radius_list = [1,2,4]

for length in length_list:
    for radius in radius_list:
        result_dir = Path(f"ecfp_{length}_{radius}")
        time_data = np.load(result_dir / Path("time.npy"))
        power_summary = pd.read_csv(result_dir / Path(f"{host}.var_monitor.dat"))

        socket_0_power = power_summary["Socket_0 Power (W)"].mean()
        socket_1_power = power_summary["Socket_1 Power (W)"].mean()
        avg_power = np.sum([socket_0_power, socket_1_power])
        print(f"{length}-{radius}\t{time_data[0]}\t{time_data[1]}\t{avg_power}\t{time_data[1] * avg_power}")