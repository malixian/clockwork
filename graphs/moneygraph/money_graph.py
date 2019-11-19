#!/usr/bin/python3
import sys
import numpy as np
import csv
from matplotlib import rcParams
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
plt.style.use('seaborn-muted')


# FONT_FAMILY = "Baskerville"
FONT_FAMILY = "Gill Sans MT Pro"

file_name = 'money_graph.pdf'

# matplotlib.font_manager._rebuild()
plt.rc('pdf', fonttype=42)
plt.rc('ps', fonttype=42)


matplotlib.rcParams['font.family'] = FONT_FAMILY
matplotlib.rcParams['font.size'] = 10
rcParams["legend.loc"] = 'best'

x_data = [0.0001, 0.001, 0.01, 0.1, 1, 10,
          100, 1000, 10000, 100000]  # per second


systems = ["google_cpu"] # amazon_gpu
systems_name = [ "Google ML-Engine CPU", "Amazon SageMaker", "Ideal"]
system_color = ['blue', 'orange', 'green', 'yellow']
system_marker = ["x", "D", "", "v", "o"]
system_linestyle = ["-", "-", "--", "-", "-"]
# system_types = ["gpu"]
graph_types = ["cost", "latency"]

data = {}

data_dir_path = "./data/money_graph/"

i = 0
for system in systems:
    # for system_type in system_types:
    for graph_type in graph_types:
        if graph_type not in data.keys():
            data[graph_type] = {}

        graph_name = "_".join([system, graph_type])
        file_name = data_dir_path + graph_name + ".csv"

        print(file_name, "\n")
        data[graph_type][graph_name] = {}
        data[graph_type][graph_name]['name'] = systems_name[i]
        data[graph_type][graph_name]['marker'] = system_marker[i]
        data[graph_type][graph_name]['color'] = system_color[i]
        data[graph_type][graph_name]['linestyle'] = system_linestyle[i]
        data[graph_type][graph_name]["x"] = []
        data[graph_type][graph_name]["y"] = []

        with open(file_name, 'r') as csvfile:
            row_reader = csv.DictReader(csvfile, delimiter=',')
            for row in row_reader:
                rate = float(row["rate"])
                cost = float(row["cost"])
                if graph_type == "cost":
                    cost *= (1000000.0 / 100.0) # 1000,000 for million requests / 100 cents for dollar
                data[graph_type][graph_name]["x"].append(rate)
                data[graph_type][graph_name]["y"].append(cost)
    i += 1
    


fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(5,3.5))


# ----------- COST

ax1.set_ylabel('cost / inf (cents)', fontname=FONT_FAMILY, fontsize=10)
ax1.set_ylim(0.001, 100000)
ax1.set_xscale('log')
ax1.set_yscale('log')
# ax1.grid(linestyle=':', linewidth='1', color='#e2e2e2')

# ax1.spines['right'].set_visible(False)
# ax1.spines['top'].set_visible(False)

# print (data)
for graph_name in data["cost"]:
        # print (data["cost"][graph_name])
    x_data = data["cost"][graph_name]['x']
    y_data = data["cost"][graph_name]['y']

    ax1.plot(x_data, y_data, label=data['cost'][graph_name]['name'], marker=data['cost'][graph_name]['marker'], color=data['cost'][graph_name]['color'], linestyle=data['cost'][graph_name]['linestyle'])
    # leg = ax1.legend(frameon=False)



# ----------- LATENCY
ax2.set_ylabel('latency / inf (ms)', fontname=FONT_FAMILY, fontsize=10)
ax2.set_xlabel('request rate (req/s)', fontname=FONT_FAMILY, fontsize=10)

ax2.set_xlim(min(x_data), max(x_data))
# ax2.set_yscale('log')

ax2.set_ylim(0, 200)
# ax2.grid(linestyle=':', linewidth='1', color='#e2e2e2')
# ax2.spines['right'].set_visible(False)
# ax2.spines['top'].set_visible(False)

# print (data)
for graph_name in data["latency"]:
        # print (data["cost"][graph_name])
    x_data = data["latency"][graph_name]['x']
    y_data = data["latency"][graph_name]['y']

    ax2.plot(x_data, y_data, label=data['latency'][graph_name]['name'], marker=data['latency'][graph_name]['marker'], color=data['latency'][graph_name]['color'], linestyle=data['latency'][graph_name]['linestyle'])
    # leg = ax2.legend(frameon=False)


    #TODO: uncomment if want the legend
    # ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.4),
    #       ncol=2, fancybox=False, shadow=False,frameon=False)





# ax1.grid(which='both', color='#f5f5f5')
# ax2.grid(which='both', color='#f5f5f5')

fig.subplots_adjust(hspace=0.1)

# plt.margins(0)
plt.tight_layout()
plt.savefig('money_graph.pdf')
plt.show()


# print X_relative
