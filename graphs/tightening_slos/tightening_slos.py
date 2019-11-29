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

fig, ax1 = plt.subplots(1,1,figsize=(5,3))


data_x = [5,7,10,15,20,30,40,50,100,150,200]
batchsizes = [1,2,4,8,16]
data = {}

with open(input_file_name, 'r') as csvfile:
    row_reader = csv.DictReader(csvfile, delimiter=',')
    for row in row_reader:
        data[str(row["batchsize"])] = [ float(row[str(x)]) for x in data_x ]
        print (data[row["batchsize"]], "\n")
        # for x in data_x:
        #     print(" -- ", x, " -- " , float(row[str(x)]))
        #     data[x] = float(row[str(x)])
        
# data = sorted(data)

# print (sorted(data.items()))

colorset = ["blue","orange","green", "red", "purple"]
markerset = ["o","v","s", "x", "+"]


i = 0
for batch_size in data.keys():
    ax1.plot( data_x , data[batch_size], label="batch size = " + str(batch_size), marker=markerset[i], color=colorset[i]) 
    i += 1


ax1.set_ylabel("System's Goodput (inference/s)", fontname=FONT_FAMILY, fontsize=10)
ax1.set_xlabel('SLO (ms)', fontname=FONT_FAMILY, fontsize=10)

ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.3),
        ncol=3, fancybox=False, shadow=False,frameon=False)

ax1.set_ylim(0)
ax1.set_xlim(0)
# ax2.set_ylim(0,1.1)

fig.tight_layout()
plt.savefig(output_file_name)

plt.show()
