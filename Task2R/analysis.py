import matplotlib.pyplot as plt
import numpy as np 
import pickle
import os

os.makedirs("./figure/", exist_ok=True)
os.makedirs("./preserve/summary/", exist_ok=True)

epoch = 180000
interval = 3000
c_stnm = 50      #the number of consuctive stimulus
stnm = int(epoch/(interval))*c_stnm


def first(Mt,Mi):
    spk = np.array(list(set(np.array(Mt)*np.array(Mi==1))))*1000
    spk = np.sort(spk)
    fst = np.ones((len(trialtm),1))*np.nan
    for i in range(len(trialtm)):
        count = 0
        if i < len(trialtm)-1:
            for j in range(len(spk)):
                if spk[j] >= trialtm[i] and spk[j] < trialtm[i+1]:
                    if spk[j] - trialtm[i] < 301:
                        fst[i,0] = spk[j] - trialtm[i]
                        break
        else:
            for j in range(len(spk)):
                if spk[j] >= trialtm[i] and spk[j] < epoch:
                    if spk[j] - trialtm[i] < 301:
                        fst[i,0] = spk[j] - trialtm[i]
                        break
    return fst

nmlist = ["_nobl","_nostbl","_nowvbl","_org"]
dnum = 50
trialtm = np.arange(10,stnm*interval/c_stnm,interval)
master = np.zeros((len(nmlist),dnum,len(trialtm)))*np.nan
master3 = np.zeros((len(nmlist), dnum, len(trialtm)))*np.nan

rng  = (0, len(trialtm))

fspk = np.ones((len(nmlist),dnum, len(trialtm),1))*np.nan


for n,dname in enumerate(nmlist):
    print(dname)
    plt.figure(figsize=(10,18))
    plt.subplots_adjust(wspace=0.2,hspace=0.3)
    for i in range(dnum):
        try:
            with open("./preserve/pthrecresult{}{:03}.pkl".format(dname,i) ,mode='rb') as f:
                weight = pickle.load(f)
                weighte_init = pickle.load(f)
                st = pickle.load(f)
                si = pickle.load(f)
                prefer = pickle.load(f)
                crr = pickle.load(f)
                Mgl = pickle.load(f)
            for t in range(len(trialtm)):
                master[n,i,t] = np.mean(crr[0,int(trialtm[t]*10):int(trialtm[t]+interval-20)*10])
            temp = np.arange(10,stnm*interval/c_stnm,interval)*10
            for j in range(len(temp)):
                master3[n,i,j] = 1e-1*np.sum(Mgl[0,int(temp[j]):int(temp[j])+interval*10-100])
            fspk[n,i] = first(np.array(st),np.array(si))
            if i < 10:
                plt.subplot(5,2,i+1)
                plt.plot(np.arange(epoch*10)/10000,crr[0,:])
                plt.title(i)
            if np.nan in crr[0,-150000:]:
                print(i,"nan included")
            print(i,len(st))
        except:
            print("{} is not working".format(i))

    plt.savefig("./figure/indivisual{}.png".format(dname), bbox_inches="tight")
    correct = (np.nanmean(master[n,:,-1])-0.5)*2

    print("correct",np.nanmean(master[n,:,:],0))
    print("first spike",np.nanmean(fspk[n,:,:,0],0))
    print("dopa", np.nanmean(master3[n],0))

master = (master - 0.5)*2

with open("./preserve/summary/pthrec_sum.pkl",mode="wb") as f:
    pickle.dump(master,f)         #successful rate
    pickle.dump(fspk,f)           #the first spike latency
    pickle.dump(master3,f)        #the amout of reward










