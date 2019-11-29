#!/usr/bin/python3
import sys
import collections
import numpy as np
import csv
from matplotlib import rcParams
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

SUCCESSFUL = 1
DROPPED = 2
FAILED = 3


file_name = "resource_utilization.csv"
output_file_name = "resource_utilization.pdf"

colorset = {0:["blue","orange","turquoise" ],1:["green", "yellow", "violet" ],2:["red", "crimson", "darkviolet"], 3: ["orange", "aqua", "lime"], 4: ["purple", "darksalmon", "palegreen"]}

plt.style.use('seaborn-muted')


# FONT_FAMILY = "Baskerville"
FONT_FAMILY = "Gill Sans MT Pro"

matplotlib.rcParams['font.family'] = FONT_FAMILY
matplotlib.rcParams['font.size'] = 10
rcParams["legend.loc"] = 'best'

# matplotlib.font_manager._rebuild()
plt.rc('pdf', fonttype=42)
plt.rc('ps', fonttype=42)


fig, (ax2, ax3, ax4) = plt.subplots(3, 1, sharex=True, figsize=(5, 5.5))



ax2.set_ylabel('utilization', fontname=FONT_FAMILY, fontsize=10)
# ax2.set_ylabel('sucess %', fontname=FONT_FAMILY, fontsize=10)
ax3.set_ylabel('drop rate (req/s)', fontname=FONT_FAMILY, fontsize=10)
ax4.set_ylabel('input (req/s)', fontname=FONT_FAMILY, fontsize=10)
ax4.set_xlabel('time (s)', fontname=FONT_FAMILY, fontsize=10)



# ax1.grid(which='both', color='#f5f5f5')
# ax2.grid(which='both', color='#f5f5f5')
# ax3.grid(which='both', color='#f5f5f5')
# ax4.grid(which='both', color='#f5f5f5')



# x_data = range(0, 1000)  # per second



data = {}

x = 0 # model_id (?)

total_data = {}
total_data["exec_latency"] = 0
total_data["queue_latency"] = 0
total_data["count"] = 0
total_data["successful"] = 0
total_data["dropped"] = {}
total_data["failed"] = {}
total_data["input"] = {}
total_data["complete"] = {}
total_data["success"] = {}
total_data["latency"] = {}





with open(file_name, 'r') as csvfile:
    row_reader = csv.DictReader(csvfile, delimiter=',')
    for row in row_reader:
        model_id = int(row["model_id"])
        x = model_id
        if model_id not in data.keys():
            data[x] = {}
            data[x]["exec_latency"] = 0
            data[x]["queue_latency"] = 0
            data[x]["count"] = 0
            data[x]["successful"] = {}
            data[x]["successful_count"] = 0
            data[x]["dropped"] = {}
            data[x]["failed"] = {}
            data[x]["input"] = {}
            data[x]["complete"] = {}
            data[x]["latency"] = {}
            data[x]["relative_successful"] = {}

        timestamp_second = int(int(row["timestamp"])/1000)
        completion_timestamp = int(row["completion_timestamp"])
        completion_second = int(completion_timestamp/1000)
        queue_time = int(row["queue_time"])
        exec_time = int(row["exec_time"])
        status = int(row["status"])
        data[x]["queue_latency"] += queue_time
        data[x]["exec_latency"] += exec_time
        data[x]["count"] += 1

        if (timestamp_second not in total_data["input"].keys()):
            total_data["input"][timestamp_second] = 0
        total_data["input"][timestamp_second] += 1

        if (completion_second not in total_data["complete"].keys()):
            total_data["complete"][completion_second] = 0
        total_data["complete"][completion_second] += 1

        if (completion_second not in data[x]["complete"].keys()):
            data[x]["complete"][completion_second] = 0
        data[x]["complete"][completion_second] += 1

        if (timestamp_second not in data[x]["input"].keys()):
            data[x]["input"][timestamp_second] = 0
            # data[x]["successful"][timestamp_second] = 0
            data[x]["dropped"][timestamp_second] = 0
            data[x]["failed"][timestamp_second] = 0
        data[x]["input"][timestamp_second] += 1

        if status == SUCCESSFUL:
            if (completion_second not in data[x]["successful"].keys()):
                data[x]["successful"][completion_second] = 0
                data[x]["latency"][completion_second] = 0
            data[x]["successful"][completion_second] += 1
            data[x]["latency"][completion_second] += queue_time + exec_time
            data[x]["successful_count"] += 1
            total_data["successful"] += 1

            if (completion_second not in total_data["success"].keys()):
                total_data["success"][completion_second] = 0
            total_data["success"][completion_second] += 1

        elif status == DROPPED:
            if (completion_second not in data[x]["dropped"].keys()):
                data[x]["dropped"][completion_second] = 0
            data[x]["dropped"][completion_second] += 1
        elif status == FAILED:
            if (completion_second not in data[x]["failed"].keys()):
                data[x]["failed"][completion_second] = 0
            data[x]["failed"][completion_second] += 1
    
    max_latency = 0            
    for x in data.keys():        
        if len(data[x]["successful"].values()) > 0 :
            data[x]["avg_throughput"] = sum(
                data[x]["successful"].values()) / len(data[x]["successful"].values())
        else: 
            data[x]["avg_throughput"] = 0
        data[x]["queue_latency"] = data[x]["queue_latency"] / data[x]["count"]
        data[x]["exec_latency"] = data[x]["exec_latency"] / data[x]["count"]

        x_data = data[x]["input"].keys()

        ordered_successful = collections.OrderedDict(
            sorted(data[x]["successful"].items()))
        ordered_dropped = collections.OrderedDict(
            sorted(data[x]["dropped"].items()))
        ordered_failed = collections.OrderedDict(sorted(data[x]["failed"].items()))        
        
        for i in ordered_successful.keys():
            data[x]["relative_successful"][i] = 100 * data[x]["successful"][i] / data[x]["complete"][i]
        ordered_relative_successful = collections.OrderedDict(sorted(data[x]["relative_successful"].items()))
        
        # ax2.plot(ordered_relative_successful.keys(), ordered_relative_successful.values(),
        #         label="model_" + str(x), marker="", color=colorset[x][0], linestyle="-")
        
        # ax2.axhline(y=[ 100 * data[x]["successful_count"]/data[x]["count"]], color=colorset[x][1], linestyle='--', label="model_" + str(x) + " success_ratio="+ str('{0:.3g}'.format(data[x]["successful_count"]/data[x]["count"])))
        ax2.set_ylim(0, 1.1)    

        ax3.plot(ordered_successful.keys(), ordered_successful.values(),
                label="successful_" + str(x), marker="", color=colorset[x][0], linestyle="-")
        # ax3.plot(ordered_dropped.keys(), ordered_dropped.values(),
        #         label="dropped_" + str(x), marker="", color=colorset[x][1], linestyle="-.")

        # ax3.plot(ordered_failed.keys(), ordered_failed.values(),
        #         label="failed_" + str(x), marker="", color=colorset[x][2], linestyle="--")
        # ax3.set_ylim(0, max(data[x]["input"].values())* 1.1)
        


        ordered_input = collections.OrderedDict(sorted(data[x]["input"].items()))
        ax4.plot(ordered_input.keys(), ordered_input.values(),
                label="Model "+str(x) , marker="", color=colorset[x][0], linestyle="-")
        ax4.set_ylim(0, max(data[x]["input"].values())* 1.1)
        
        ordered_latency = collections.OrderedDict(sorted(data[x]["latency"].items()))  

        ax2.plot(ordered_successful.keys(),np.divide(list(ordered_latency.values()), list(ordered_successful.values())), label="model_"+str(x), marker="", color=colorset[x][0], linestyle="-")
        ax2.set_ylim(0)
        print(x, data[x]["avg_throughput"],  data[x]
            ["queue_latency"], data[x]["exec_latency"])
    
    if (len(data.keys()) > 1):        
        ordered_total = collections.OrderedDict(
            sorted(total_data["input"].items()))
        ordered_total_througput = collections.OrderedDict(
                sorted(total_data["success"].items()))
        ax4.plot(ordered_total.keys(), ordered_total.values(),
        label="Total" , marker="", color="slategray", linestyle=":", alpha=0.6)
        ax3.plot(ordered_total_througput.keys(), ordered_total_througput.values(),
        label="system throughput [" + str(total_data["successful"]) + "]" , marker="", color="slategray", linestyle=":", alpha=0.6)
        ax4.set_ylim(0, max(ordered_total.values())* 1.1)    
        ax3.set_ylim(0, max(ordered_total.values())* 1.1)    

        # ax3.set_ylim(0, 1500)
        # ax4.set_ylim(0, 1500)

sum_success_ratio = 0
sum_squared_success_ratio = 0

for x in data.keys():
    sum_success_ratio += data[x]["successful_count"]/data[x]["count"]    
    sum_squared_success_ratio += (data[x]["successful_count"]/data[x]["count"]) ** 2


# jain_index = (float) (sum_success_ratio ** 2) / ( len(data.keys()) * sum_squared_success_ratio )
# rezas_index = (float) (sum_success_ratio) / ( len(data.keys())) * jain_index
# avg_success_ratio = (float) (sum_success_ratio) / ( len(data.keys()))

# print (plot_title, "'s jain's fairness index : ", jain_index)
# print (plot_title, "'s reza's fairness index : ", rezas_index)
# print (plot_title, "'s reza's avg_success_ratio : ", avg_success_ratio)


# ax1.legend(frameon=False)
# ax2.legend(frameon=True)
# ax3.legend(frameon=True)


ax4.legend(loc='upper center', bbox_to_anchor=(0.5, -0.4),
        ncol=3, fancybox=False, shadow=False,frameon=False)


fig.subplots_adjust(hspace=0.05)
ax3.set_xlim(min(x_data), max(x_data))



ax2.set_title("(c) Resource Utilization")
ax3.set_title("(b) Request Drop Rate")
ax4.set_title("(a) Input")


plt.tight_layout()
plt.margins(0)

plt.savefig(output_file_name)
plt.show()

