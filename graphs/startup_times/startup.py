#!/usr/bin/python3
import sys
import numpy as np
import csv
from matplotlib import rcParams
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
plt.style.use('seaborn-muted')

input_file_name = 'tightening_data.csv'
output_file_name = "tightning.pdf"


# FONT_FAMILY = "Baskerville"
FONT_FAMILY = "Gill Sans MT Pro"

schedulers = ["clockwork","wfq","rand","cr","edf","fifo"]
scheduler_names = {"clockwork":"Clockwork","wfq":"WFQ","rand":"Rand","cr":"CR","edf":"EDF","fifo":"FIFO"}

# matplotlib.font_manager._rebuild()
plt.rc('pdf', fonttype=42)
plt.rc('ps', fonttype=42)


matplotlib.rcParams['font.family'] = FONT_FAMILY
matplotlib.rcParams['font.size'] = 10
rcParams["legend.loc"] = 'best'


# matplotlib.font_manager._rebuild()
plt.rc('pdf', fonttype=42)
plt.rc('ps', fonttype=42)

countries = ['Clockwork', 'BARISTA']
vm_spawn = np.array([0, 300])
container_download = np.array([0, 46])
model_load = np.array([11, 125])
inference = np.array([5, 5])
ind = [x for x, _ in enumerate(countries)]

fig, ax1 = plt.subplots(1,1,figsize=(5,1.55))

ax1.barh(ind, inference, height=0.3, label='inference', color='red', left=vm_spawn+container_download+model_load)
ax1.barh(ind, model_load, height=0.3, label='model load', color='green', left=vm_spawn+container_download)
ax1.barh(ind, container_download, height=0.3, label='container download', color='orange', left=vm_spawn)
ax1.barh(ind, vm_spawn, height=0.3, label='vm spawn', color='blue')

plt.yticks(ind, countries)
ax1.set_xlabel("latency (ms)")
# ax1.set_ylabel("Countries")
ax1.legend(loc="lower right", fancybox=False, shadow=False,frameon=False, ncol=2)


rects = ax1.patches

# For each bar: Place a label

offset = {}

matplotlib.rcParams['font.size'] = 8

for i in range(len(rects)-1, -1, -1):

    rect = rects[i]
    # Get X and Y placement of label from rect.
    x_value = rect.get_width()
    y_value = rect.get_y() + rect.get_height() / 2

    if y_value not in offset.keys():
        offset[y_value] = 0

  
    # Number of points between bar and label. Change to your liking.
    space = 0
    # Vertical alignment for positive values
    ha = 'center'

    if x_value == 0:
        continue

    # If value of bar is negative: Place label left of bar
    if x_value < 0:
        # Invert space to place label to the left
        space *= -1
        # Horizontally align label at right
        ha = 'center'

    # Use X value as label and format number with one decimal place
    label = x_value

    adjust = 0
    
    # if offset[y_value] != 0 :
    #     adjust = 3
    # else:
    #     adjust = 1

    # Create annotation
    plt.annotate(
        label,                      # Use `label` as label
        ((x_value / 2.0) + offset[y_value] + adjust, y_value),         # Place label at end of the bar
        xytext=(space, 0),          # Horizontally shift label by `space`
        textcoords="offset points", # Interpret `xytext` as offset in points
        va='center',                # Vertically center label
        color='white',
        ha=ha)                      # Horizontally align label differently for
                                    # posit

    offset[y_value] += x_value 

plt.yticks(rotation=90)
fig.tight_layout()

plt.savefig("startup.pdf")

plt.show()
