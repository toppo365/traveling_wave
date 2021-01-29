#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np 
import pickle
import os

os.makedirs("./figure/", exist_ok=True)
os.makedirs("./preserve/summary/", exist_ok=True)

dnum = 50
dur = 3000000
stnm = 100
chance = 0.5
interval = 3000
term = np.arange(10,(stnm+1)*interval,interval)*10

nmlist = ["_nobl","_nostbl","_nowvbl","_org"]

allprefer = np.zeros((len(nmlist),dnum,int(stnm/4)))

for n,dname in enumerate(nmlist):
    print(dname)
    correct = []
    pref = np.ones((dnum,stnm))*chance
    pref2 = np.ones((dnum,int(stnm/4)))*np.nan
    plt.figure(figsize=(20,15))
    plt.subplots_adjust(wspace=0.2,hspace=0.3)
    for i in range(dnum):
        try:
            with open("./preserve/rnxrresult{}{:03}.pkl".format(dname,i) ,mode='rb') as f:
                weight = pickle.load(f)
                weighte_init = pickle.load(f)
                st = pickle.load(f)
                si = pickle.load(f)
                prefer = pickle.load(f)
                crr = pickle.load(f)
            prefer = np.array(prefer)
            prefer[prefer==0.5] = chance
            for j in range(stnm):
                pref[i,j] = prefer[0,int(term[j])+interval*10-110]
            pref2[i,:] = [np.mean(np.array(pref[i,4*k:4*(k+1)])) for k in range(int(stnm/4))]
            if i < 10:
                plt.subplot(5,4,2*i+1)
                plt.plot(np.arange(dur)/10000,prefer[0,:])
                plt.title(i)
                plt.xlabel("time[s]")
                plt.subplot(5,4,2*i+2)
                plt.plot(np.arange(dur)/10000,crr[0,:])
                plt.title(i)
                plt.xlabel("time[s]")
            print(i)
        except:
            print("{} is not working".format(i))

    plt.savefig("./figure/indivisual{}.png".format(dname), bbox_inches="tight")
    allprefer[n] = pref2
    pref2 = np.nanmean(pref2,axis = 0)
    print(pref2)

with open("./preserve/summary/rnxr_sum.pkl",mode="wb") as f:
    pickle.dump(allprefer,f)    #successful rate


