#!/usr/bin/python

##################################################
#
# Generic plotting script 
# Author: Colin de Vrieze <cdv@berkeley.edu>
#
##################################################

##################################################
# Imports

import matplotlib.pyplot as plt
import numpy as np
import csv
import argparse
from matplotlib2tikz import save as tikz_save

##################################################
# parameters

colors = ['#3D9970', '#85144b','#FF4136', '#0074D9', '#FFDC00']

SMALL_SIZE = 14
MEDIUM_SIZE = 18 
BIGGER_SIZE = 20 

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize

##################################################
# read arguments

parser = argparse.ArgumentParser()

parser.add_argument("file", type=str, nargs='*',
           help="path to csv file()")
parser.add_argument('-num', action='store_true', help="plot only number of clusters")
parser.add_argument("--title", type=str, default="Generic Plot",
           help="title to be shown in plot")


args = parser.parse_args()
if (len(args.file) == 2):
    double_y = True
    csvfile2 = args.file[1]
else:
    double_y = False

csvfile = args.file[0]
title = args.title

##################################################
# load data from csv files

with open(csvfile,'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    labels = next(reader)

    cols = len(labels)
    data = [[] for j in range(cols)] 

    for (i,row) in enumerate(reader):
        for k in range(cols):
            data[k].append(float(row[k]))

    for j in range(cols):
       data[j] = np.array(data[j]) 

    print("Read %d lines in %d colums" % (i+1,cols))

if (double_y): # read data for second y axis

    with open(csvfile2,'r') as csvfile2:
        reader2 = csv.reader(csvfile2, delimiter=',')
        labels2 = next(reader2)
        cols2 = len(labels2)
        data2 = [[] for j in range(cols2)] 

        for (i,row) in enumerate(reader2):
            for k in range(cols2):
                data2[k].append(float(row[k]))

        for j in range(cols2):
           data2[j] = np.array(data2[j]) 

        print("Read %d lines in %d colums" % (i+1,cols2))

##################################################
# plotting

fig = plt.figure()
plt.title(title, fontsize=BIGGER_SIZE)
ax = fig.add_subplot(111)
plt.show(block=False)

# ax.set_xlim([0, 16]) 
ax.set_xlim([0, 1000]) 
h, l = [], [] # legend handels


if (double_y): # draw second y axis
    ax2 = ax.twinx()
    ax2.set(ylabel='Number of Clusters')
    ax2.set_ylim([0, 25])
    ax2.set_xlim([0, 16])

    for k in np.arange(1,len(data2),2):
        color = colors.pop()
        ax2.plot(data2[0], data2[k], color=color, label=labels2[k], lw=3)
        ax2.fill_between(data2[0], data2[k]-data2[k+1]/2.0, data2[k]+data2[k+1]/2.0,
        alpha=0.35, edgecolor=color, facecolor=color, lw=0)

    h2, l2 = ax2.get_legend_handles_labels()    

else:
    colors.pop()

if (not args.num): # plot BER
    ax.set(ylabel='Bit-Error Rate (BER)', xlabel='$E_b/N_0$ [dB]')
    ax.set_ylim([0, 1])
    # ax.set_ylim([1e-7, 1])
    # ax.set_yscale('log')
else: # plot number of clusters 
    ax.set(ylabel='Number of clusters', xlabel='$E_b/N_0$ [dB]')
    ax.set_ylim([0, 25])


for i in np.arange(1,len(data),2):
    print(i)
    color = colors.pop()
    ax.plot(data[0], data[i], color=color, label=labels[i], lw=3)
    ax.fill_between(data[0], data[i]-data[i+1]/2.0, data[i]+data[i+1]/2.0,
    alpha=0.35, edgecolor=color, facecolor=color, lw=0)

h1, l1 = ax.get_legend_handles_labels()    

if (double_y):
    h, l = h1+h2, l1+l2
else:
    h, l = h1, l1

ax.legend(h, l,loc=0)

plt.draw()
fig.savefig("output.png")
tikz_save("output.tex", figureheight='2in', figurewidth='3.4in')
plt.show()
