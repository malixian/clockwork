#!/usr/bin/python3
import sys
import numpy as np
import csv
from matplotlib import rcParams
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
plt.style.use('seaborn-muted')

fairness_input_file_name = 'fairness_data.csv'
success_ratio_fairness_input_file_name = 'avg_success_ratio.csv'
output_file_name = "fairness.pdf"


# FONT_FAMILY = "Baskerville"
FONT_FAMILY = "Gill Sans MT Pro"



# matplotlib.font_manager._rebuild()
plt.rc('pdf', fonttype=42)
plt.rc('ps', fonttype=42)


matplotlib.rcParams['font.family'] = FONT_FAMILY
matplotlib.rcParams['font.size'] = 10
rcParams["legend.loc"] = 'best'


# matplotlib.font_manager._rebuild()
plt.rc('pdf', fonttype=42)
plt.rc('ps', fonttype=42)

fig, (ax1, ax2) = plt.subplots(1,2,figsize=(5,2.5))


schedulers = ["clockwork","wfq","rand","cr","edf","fifo"]
scheduler_names = {"clockwork":"Clockwork","wfq":"WFQ","rand":"Rand","cr":"CR","edf":"EDF","fifo":"FIFO"}
data = {}
normalized_data = {}


avg_success_ratio = {}
normalized_avg_success_ratio = {}


for scheduler in schedulers:
    data[scheduler] = []
    normalized_data[scheduler] = []
    avg_success_ratio[scheduler] = []
    normalized_avg_success_ratio[scheduler] = []

with open(fairness_input_file_name, 'r') as csvfile:
    row_reader = csv.DictReader(csvfile, delimiter=',')
    for row in row_reader:
        for scheduler in schedulers:
            data[scheduler].append(float(row[scheduler]))
            normalized_data[scheduler].append(float(row[scheduler]) / max([float(row[s]) for s in schedulers]))


with open(success_ratio_fairness_input_file_name, 'r') as csvfile:
    row_reader = csv.DictReader(csvfile, delimiter=',')
    for row in row_reader:
        for scheduler in schedulers:
            avg_success_ratio[scheduler].append(float(row[scheduler]))
            normalized_avg_success_ratio[scheduler].append(float(row[scheduler]) / max([float(row[s]) for s in schedulers]))


# data = sorted(data)
# normalized_data = sorted(normalized_data)

# print ([x for x in data.values()])

ax1.boxplot([x for x in normalized_data.values()])
ax2.boxplot([x for x in normalized_avg_success_ratio.values()])

ax1.set_title("Normalized Fairness Index")
ax2.set_title("Normalized Avg Success Ratio")


# colors = ["blue","orange", "yellow", "violet" ,"brown", "crimson", "darkviolet","steelblue", "aqua", "lime"]

# for patch, color in zip(bplot['boxes'], colors):
#     patch.set_facecolor(color)

# ax.set_title("Box Plot")

plt.setp(ax1, xticks=[x for x in range (1, len(normalized_data.keys())+1)], xticklabels=[scheduler_names[s] for s in normalized_data.keys()])
plt.setp(ax2, xticks=[x for x in range (1, len(normalized_avg_success_ratio.keys())+1)], xticklabels=[scheduler_names[s] for s in normalized_avg_success_ratio.keys()])

# ax1.set_xticklabels([x for x in range (1, len(data.keys())+1)], [scheduler_names[s] for s in data.keys()])
# ax2.set_xticklabels([x for x in range (1, len(data.keys())+1)], [scheduler_names[s] for s in data.keys()])

fig.autofmt_xdate()

fig.tight_layout()


ax1.set_ylim(0,1.1)
ax2.set_ylim(0,1.1)

plt.savefig(output_file_name)

plt.show()
