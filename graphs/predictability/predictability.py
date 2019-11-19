#!/usr/bin/python3
import sys
import numpy as np
import csv
from matplotlib import rcParams
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
plt.style.use('seaborn-muted')

inference_predictability_input_file_name = 'inference_predictability.csv'
model_load_predictability_input_file_name = 'model_load_predictability.csv'
output_file_name = "predictability.pdf"


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


models = ["resnet18","resnet50","resnet152"]
model_names = {"resnet18":"Resnet18","resnet50":"Resnet50","resnet152":"Resnet152", "googlenet":"GoogleNet", "alexnet":"Alexnet","vgg16":"VGG16"}

inference_latency = {}
model_load_latency = {}

for model in models:
    inference_latency[model] = []
    model_load_latency[model] = []

with open(inference_predictability_input_file_name, 'r') as csvfile:
    row_reader = csv.DictReader(csvfile, delimiter=',')
    for row in row_reader:
        for model in models:
            inference_latency[model].append(float(row[model]))


with open(model_load_predictability_input_file_name, 'r') as csvfile:
    row_reader = csv.DictReader(csvfile, delimiter=',')
    for row in row_reader:
        for model in models:
            model_load_latency[model].append(float(row[model]))

# data = sorted(data)
# normalized_data = sorted(normalized_data)

# print ([x for x in data.values()])

ax1.boxplot([x for x in [inference_latency[k] for k in models] ])
ax2.boxplot([x for x in [model_load_latency[k] for k in models] ])

ax1.set_title("Inference")
ax2.set_title("Model Load")


ax1.set_ylabel('Latency (ms)', fontname=FONT_FAMILY, fontsize=10)
ax2.set_ylabel('Latency (ms)', fontname=FONT_FAMILY, fontsize=10)

# colors = ["blue","orange", "yellow", "violet" ,"brown", "crimson", "darkviolet","steelblue", "aqua", "lime"]

# for patch, color in zip(bplot['boxes'], colors):
#     patch.set_facecolor(color)

# ax.set_title("Box Plot")

plt.setp(ax1, xticks=[x for x in range (1, len(inference_latency.keys())+1)], xticklabels=[model_names[s] for s in models])
plt.setp(ax2, xticks=[x for x in range (1, len(model_load_latency.keys())+1)], xticklabels=[model_names[s] for s in models])

# ax1.set_xticklabels([x for x in range (1, len(data.keys())+1)], [scheduler_names[s] for s in data.keys()])
# ax2.set_xticklabels([x for x in range (1, len(data.keys())+1)], [scheduler_names[s] for s in data.keys()])

fig.autofmt_xdate()

fig.tight_layout()


ax1.set_ylim(0)
ax2.set_ylim(0)

plt.savefig(output_file_name)

plt.show()
