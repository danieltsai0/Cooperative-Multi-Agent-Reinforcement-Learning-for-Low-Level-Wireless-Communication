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

##################################################
# parameters

colors = ['#0074D9', '#3D9970', '#85144b','#FF4136']

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

with open(csvfile,'rb') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    labels = reader.next()
    cols = len(labels)
    data = [[] for j in range(cols)] 

    for (i,row) in enumerate(reader):
        for k in range(cols):
            data[k].append(float(row[k]))

    for j in range(cols):
       data[j] = np.array(data[j]) 

    print("Read %d lines in %d colums" % (i+1,cols))

if (double_y): # read data for second y axis

    with open(csvfile2,'rb') as csvfile2:
        reader2 = csv.reader(csvfile2, delimiter=',')
        labels2 = reader2.next()
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

ax.set(ylabel='Bit-Error Rate (BER)', xlabel='$E_b/N_0$ [dB]')
ax.set_yscale('log')
ax.set_xlim([-10, 8])
ax.set_ylim([1e-7, 1])

if (double_y): # draw second y axis
    ax2 = ax.twinx()
    ax2.set(ylabel='Number of Clusters')
    ax2.set_ylim([0, 20])
    ax2.set_xlim([-10, 8])

    for k,col in enumerate(data2):
        if (k==0): continue
        ax2.plot(data2[0], data2[k], color=colors.pop(), lw=3, alpha=0.2)

for i,col in enumerate(data):
    if (i==0): continue
    ax.plot(data[0], data[i], color=colors.pop(), label=labels[i], lw=3)

ax.legend(loc=0)

plt.draw()
plt.show()
