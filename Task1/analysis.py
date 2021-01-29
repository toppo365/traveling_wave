#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import numpy as np 
import pickle
import os

os.makedirs("./figure/", exist_ok=True)
os.makedirs("./preserve/summary/", exist_ok=True)

dnum = 50
dur = 2400000
stnm = 80
chance = 0.25
interval = 3000
term = np.arange(10,(stnm+1)*interval,interval)*10

nmlist = ["_nobl","_nostbl","_nowvbl","_org"]

allprefer = np.zeros((len(nmlist),dnum,int(stnm)))

for n,dname in enumerate(nmlist):
    print(dname)
    correct = []
    pref = np.ones((dnum,stnm))*np.nan
    plt.figure(figsize=(10,15))
    plt.subplots_adjust(wspace=0.2,hspace=0.3)
    for i in range(dnum):
        try:
            with open("./preserve/rn14result{}{:03}.pkl".format(dname,i) ,mode='rb') as f:
                weight = pickle.load(f)
                weighte_init = pickle.load(f)
                st = pickle.load(f)
                si = pickle.load(f)
                prefer = pickle.load(f)
                crr = pickle.load(f)
            prefer = np.array(prefer)
            prefer[prefer==0.5] = chance
            for j in range(stnm):
                pref[i,j] = np.where(np.mean(prefer[0,int(term[j])+100:int(term[j+1])]) < chance, 0, 1)
            if i < 10:
                plt.subplot(5,2,i+1)
                plt.plot(np.arange(dur)/10000,prefer[0,:dur])
                plt.xlabel("time[s]")
                plt.title(i)
            correct.append(np.mean(pref[i,-6:]))
            print(i)
        except:
            print("{} is not working".format(i))
    plt.savefig("./figure/indivisual{}.png".format(dname), bbox_inches="tight")
    correct = np.array(correct)
    allprefer[n] = pref


with open("./preserve/summary/rn14_sum.pkl",mode="wb") as f:
    pickle.dump(allprefer,f)    #successful rate





