#!/usr/bin/python3
import sys
import numpy as np
import csv
from matplotlib import rcParams
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
plt.style.use('seaborn-muted')

input_file_name = 'batch_size_inference.csv'
output_file_name = "batch_size_inference.pdf"


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

fig, ax1 = plt.subplots(1,1,figsize=(5,3))


batch_sizes = [1,2,4,8,16]
data = {}

with open(input_file_name, 'r') as csvfile:
    row_reader = csv.DictReader(csvfile, delimiter=',')
    for row in row_reader:
        data[row["title"]] = [ float(row['1']), float(row['2']), float(row['4']), float(row['8']), float(row['16'])]
        
data = sorted(data.items())

# print (sorted(data.items()))

colorset = ["blue","orange","green", "red", "purple"]
markerset = ["o","v","s", "x", "+"]


i = 0
for graph_name, values in data:
    ax1.plot( [ str(x) for x in batch_sizes] , values, label=graph_name, marker=markerset[i], color=colorset[i]) 
    i += 1



# ax1.set_title("(a) Jain's Fairness Index")



# colors = ["blue","orange", "yellow", "violet" ,"brown", "crimson", "darkviolet","steelblue", "aqua", "lime"]

# for patch, color in zip(bplot['boxes'], colors):
#     patch.set_facecolor(color)

# ax.set_title("Box Plot")

# plt.setp(ax1, xticks=[x for x in range (1, len(data.keys())+1)], xticklabels=[da[s] for s in data.keys()])
# plt.setp(ax2, xticks=[x for x in range (1, len(data.keys())+1)], xticklabels=[scheduler_names[s] for s in data.keys()])

# ax1.set_xticklabels([x for x in range (1, len(data.keys())+1)], [scheduler_names[s] for s in data.keys()])
# ax2.set_xticklabels([x for x in range (1, len(data.keys())+1)], [scheduler_names[s] for s in data.keys()])

# fig.autofmt_xdate()



ax1.set_ylabel("Inference Latency (ms)", fontname=FONT_FAMILY, fontsize=10)
ax1.set_xlabel('Batch Size', fontname=FONT_FAMILY, fontsize=10)

ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25),
        ncol=3, fancybox=False, shadow=False,frameon=False)

ax1.set_ylim(0)
ax1.set_xlim(-0.25)
# ax2.set_ylim(0,1.1)

plt.margins(0)

fig.tight_layout()
plt.savefig(output_file_name)

plt.show()
