#!/usr/bin/env python
# coding: utf-8

from brian2 import *
import random
import copy
import pickle
import sys


class ProgressBar(object):
    def __init__(self, toolbar_width=20):
        self.toolbar_width = toolbar_width
        self.ticks = 0

    def __call__(self, elapsed, complete, start, duration):
        if complete == 0.0:
            # setup toolbar
            sys.stdout.write("[%s]" % ("" * self.toolbar_width))
            sys.stdout.flush()
            sys.stdout.write("\b" * (self.toolbar_width + 1)) # return to start of line, after '['
        else:
            ticks_needed = int(round(complete * self.toolbar_width))
            if self.ticks < ticks_needed:
                sys.stdout.write("-" * (ticks_needed-self.ticks))
                sys.stdout.flush()
                self.ticks = ticks_needed
        if complete == 1.0:
            sys.stdout.write("]\n")


row = 1
column = 120   
N = row*column + 6
rng = 1
target = [4+i for i in range(2)]

temp = [i for i in range(N) if ((i % column)%2 == 1) and (int(i/column) % 2 == 1)]
Ne = N

nrny = [0,0,0,0,row+1,row+1]
nrny.extend([1 if i < column else 2 for i in range(column*row)])
nrnx = [15,45,75,105,30,90]
nrnx.extend([i % column for i in range(column*row)])
nrnx = np.array(nrnx)
nrny = np.array(nrny)

temp = []
temp2 = []

for i in range(target[-1]+1,N):
    r = int((i - target[-1] - 1) / 30)
    temp.append((r,i))
    if i % 2 == 0:
        temp.append((i,4))
    else:
        temp.append((i,5))
    cand = [x for x in range(4) if x != r]
    cand = np.random.permutation(array(cand))
    temp.append((int(cand[0]),i))
    
temp = sorted(temp)
temp = np.array([list(i) for i in temp])

temp2 = []
ii1 = []
jj1 = []
sss = []
for i in range(len(temp[:,1])):
    ii1.append(temp[i,0])
    jj1.append(temp[i,1])
    if temp[i,0] < target[0] and temp[i,0] != int((temp[i,1] - target[-1] - 1) / 30):
        sss.append(0.1)
    else:
        sss.append(0.2)



epoch = 300000
simulation_duration = epoch* ms

## Neurons
taum = 10*ms
tausp = 10*ms
Ee = 0*mV
vt = -54*mV
vr = -60*mV
El = -70*mV
taue = 5*ms
taug = 5*ms
neuron_spacing = 100*umeter

## STDP
taupre = 30*ms
taupost = taupre
gmax = 0.24
gmin = 0.0
dApre = .01
dApost = -dApre * taupre / taupost * 1.05
dApost *= gmax
dApre *= gmax

delta = 4*mV

## Dopamine signaling
tauc = 1000*ms
taud = 200*ms
taus = 1*ms
epsilon_dopa = 0.002

interval = 3000
taui = 200*ms

alpha = 4
beta = 0.13*mV 
theta = 0

clcinterval = 20*ms

#inhibitory neuron eq
geqs = '''
    dsp/dt = -sp/taug : 1
    dgcl/dt = 1/(1*ms) : 1
'''

#excitatory -> inhibitory (pre)
gpre = '''
    v_post -= beta*(sp_pre - theta) 
'''

#dopamine neuron eq
deqs = '''
    count : 1
    dsp/dt = -sp/tausp : 1 
    ddopa/dt = -dopa/taui :1 
    mode : 1
    nrt : 1
    prefer : 1
    preferg : 1
    preferall : 1
    prefer1 : 1
    prefer2 : 1
    prefer3 : 1
    prefer4 : 1
    phase : 1
    crr : 1
    crr1 : 1
    crr2 : 1
    crr3 : 1
    crr4 : 1
    crow : 1
    '''

#excitatory neurons G eq
eqs = '''
    dem : 1 
    sup : 1 
    delt = clip(sup*5+3,0,6)*mV : volt
    dv/dt = (ge * (Ee - vr) + El - v) / taum + delt*sqrt(2/taum)*xi : volt
    dge/dt = -ge / taue : 1
    nrt : 1
    x : metre 
    y : metre
    rfr : second
'''

#clc eq
ceqs = '''
    dem: 1 
    dsup/dt = (dem-sup)/(200*ms) : 1 (unless refractory)
    dv/dt = 1/clcinterval : 1
    x : metre
    y : metre
    store : 1
    store2 : 1
    refac : second
    bc : 1
    bcnt : 1
'''

#synapse among excitatory neurons eq
smodel = '''mode: 1
    ratio : 1
    ratio1 : 1
    ratio2 : 1
    crr : 1
    dc/dt = -c / tauc : 1 (clock-driven)
    dd/dt = -d / taud : 1 (clock-driven)
    ds/dt = mode * c * (d *(1-(d < 0)*(c < 0)))/ taus : 1 (clock-driven)
    dApre/dt = -Apre / taupre : 1 (event-driven)
    dApost/dt = -Apost / taupost : 1 (event-driven)
'''

#wave component eq
s2model ='''
    sup -= sup*(tempest==1)*(initial!=1)
    store -= store*(tempest==1)
    bcnt -= bcnt*(tempest==1)*(initial!=1)
    tempest += int(c_stnm*((tempest < c_stnm-1)-0.5)-(c_stnm/2-1))
    dem = (2*tempest*(tempest > 1) - 5*(tempest == 0.0))
'''

#clock -> wave (pre)
cpre = '''
    bcnt2 -= bcnt2 * (bcnt_pre == 0) * (bcnt_post == 0)
    bcnt_post += (bcnt2 == 0)*((-sup_post+store_pre)>1e-3)
    bcnt2 += (bcnt2 == 0)*((-sup_post+store_pre)>1e-3)
    store2_post += 0.2*(store_pre-sup_post)*((-sup_post+store_pre)>1e-3)
    store2_pre -= 0.1*(-sup_post+store_pre)*((-sup_post+store_pre)>1e-3)
'''

#rest process of clc
rst = '''
    v = 0
    store = sup * 1
    sup += store2 * tanh(bcnt)
    sup = clip(sup,-1,100)
    store2 = 0
'''

#STDP rule pre
pre1='''
    ge += clip(s, gmin, gmax)
    Apre += dApre*(1-crr)*2
    c = clip(c + mode * Apost, -gmax/2, gmax/2)
    s = clip(s, gmin, gmax)
'''

#STDP rule post
post='''
    Apost += dApost*(1-crr)*2
    c = clip(c + mode * Apre, -gmax/2, gmax/2)
    s = clip(s, gmin, gmax)
'''

#spike generater -> G (pre)
sgpre = '''
    v += w*50*mV
    rfr = 3*interval*(tempest == 0)*ms
    tempest += (c_stnm*((tempest < c_stnm-1)-0.5)-(c_stnm/2-1))
'''

#dopamine -> exc_s (pre)
repre = '''
    mode_pre = 2*(prefer_pre-0.5)
    d_post = clip(d_post + dopa_pre*mode_pre*2,-0.3,0.3)
    nrt_pre = (1-crr_pre)
    crr_post = crr_pre
'''

#G -> dopamine (pre)
de1pre = '''
    sp_post += abs(by)
    nrt_pre = nrt_post*1
    prefer1 += int(by==1)
    prefer2 += int(by==2)
    preferall = int(prefer1 > prefer2+4) - int(prefer1 < prefer2-4)
    prefer = 0.5 + (preferall*phase_post)/2
'''

#input -> dopamine (pre)
de2pre = '''
    crr_post = clip(crr_post + ((prefer>0.5)-0.5)*(counter == 0)/5,0.5,1)
    phase_post = w
    counter += 1 - (counter+1)*int(counter == c_stnm -1)
    prefer = (prefer-0.5)*int(counter != 1)+0.5
    preferall = (preferall)*int(counter != 1)
    prefer1 = (prefer1)*int(counter != 1)
    prefer2 = (prefer2)*int(counter != 1)
    dopa_post = epsilon_dopa
    count_post += (counter == 1)*(1-4*(count_post == 4))
'''

#input -> dopamine (post)
scpre ='''
    v_post = 0.01
    bcnt_post -= bcnt_post * (sup < 1e-3)
'''



start_scope()

#excitatory neurons
G = NeuronGroup(N, eqs, threshold='v>vt', reset='v = vr', refractory='rfr', method='milstein')
G.v = "5*rand()*mV+El"
G.x = nrnx*neuron_spacing
G.y = nrny*neuron_spacing
G.dem = 0
G.sup = 0
G.nrt = 1

exc_g = G

#designing external stimuli 
c_stnm = 50  #the number of consuctive stimulus
stnm = int(epoch/(interval))*c_stnm
input_indices = zeros(stnm)
input_indices[[i for i in range(stnm) if int(i/c_stnm)%4 == 1]] = 1
input_indices[[i for i in range(stnm) if int(i/c_stnm)%4 == 2]] = 2
input_indices[[i for i in range(stnm) if int(i/c_stnm)%4 == 3]] = 3

input_times = np.array([arange(base,base+c_stnm*5,5) for base in arange(10,stnm*interval/c_stnm,interval)]).ravel()
input = SpikeGeneratorGroup(4, input_indices, input_times*ms)

#pipe from external stimulus to excitatory neurons
synapse = Synapses(input, G, model='''w:1
                                      tempest : 1''',on_pre=sgpre)
synapse.connect(i=[0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3], j=[0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3])
weak = 0.0
synapse.w = [1,1,weak,weak,1,weak,1,weak,weak,1,weak,1,weak,weak,1,1]
synapse.tempest = 1

#synapses among excitory neurons
exc_s = Synapses(exc_g, G, model=smodel,on_pre=pre1,on_post=post,delay=2*ms,method="euler")
exc_s.connect(i=ii1,j=jj1)
exc_s.d = 0
exc_crr = 0.5
exc_s.mode = 0.3  
exc_s.s = array(sss)

    
#clock
cmaster = NeuronGroup(1,'''dv/dt = 1/clcinterval : 1''', threshold = 'v > 1', reset = '''v = 0''',method="euler")
cmaster.v = 0

clc = NeuronGroup(N, ceqs, threshold='v > 1', reset=rst, refractory="refac",method='euler')
clc.x = nrnx*neuron_spacing
clc.y = nrny*neuron_spacing
clc.v = 0.01

cint = Synapses(clc, clc, model='''bcnt2 : 1''',on_pre = cpre, delay = clcinterval,method="euler")
cint.connect(i=ii1,j=jj1)
sclc = Synapses(cmaster,clc,model = '''''', on_pre = scpre)
sclc.connect(p=1)

#synapse from clock to other neurons
sups = Synapses(clc, G, model='''''',on_pre="""sup_post = sup_pre
                                               dem_post = dem_pre""", delay=1*ms, method="euler")
sups.connect(j = 'i')

#pipe from external stimulus to clock
syn2 = Synapses(input, clc, model='''tempest : 1
                                     initial : 1
                                     counter : 1''', on_pre = s2model, delay = 10*ms)
syn2.connect(i=[0,0,1,1,2,2,3,3], j=[0,1,0,2,1,3,2,3])
syn2.tempest = 1
syn2.initial = 1
stoc = Synapses(input,clc,model = '''ccnt : 1''', on_pre = '''bcnt_post -= bcnt_post*(ccnt==0)
                                                              ccnt += 1 - c_stnm*(ccnt==(c_stnm-1))''')
stoc.connect(p=1)

#the definition of global inhibitory neuron
glob = NeuronGroup(1, geqs, threshold = "gcl>=1",reset = "gcl=0", method='euler')
glob.sp = 0

#synapses from global inhibitory neuron to excitatory neuron
sglb = Synapses(glob, G, on_pre = gpre, on_post="sp_pre += 1", delay = 1*ms, method="euler")
sglb.connect()

#global reward signal
dopamine = NeuronGroup(1, deqs, threshold = "sp>0.8",reset="sp=0",method="euler")
dopamine.mode = 0
dopamine.count = 0
dopamine.nrt = 1
dopamine.preferall = 1
dopamine.crr = 0.5
detecter = Synapses(G,dopamine,model='''by : 1''', on_pre = de1pre, method='euler')     #sp increase from output neurons
detecter.connect(i=range(N),j = [0 for i in range(N)])
detecter.by[target] = [1,2]

detecter2 = Synapses(input,dopamine,model='''w :1
                                             counter : 1''',on_pre=de2pre, method='euler')             #dopamine delivering preparation
detecter2.connect(i=[0,1,2,3], j=[0,0,0,0])
detecter2.w[[0,1,2,3]] = [-1,1,1,-1]

#pipe from global reward signal to each excitatory synapse
reward = Synapses(dopamine, exc_s, model='''tempest : 1''', on_pre=repre, delay=100*ms, method='euler')
reward.connect(p=1)
reward.tempest = 0.5

weighte_init = copy.deepcopy(array(exc_s.s))


M = SpikeMonitor(G)
Mdopa = StateMonitor(dopamine, ['crr','prefer','mode','preferall','prefer1'], record=[0])

run(simulation_duration,report=ProgressBar(),report_period=1*second)



with open("./preserve/rnxrsetup_nostbl{}.pkl".format(sys.argv[1]) ,mode='wb') as f:
    pickle.dump(ii1,f)
    pickle.dump(jj1,f)
    pickle.dump(nrnx,f)
    pickle.dump(nrny,f)
    pickle.dump(input_indices,f)
    pickle.dump(input_times,f)


with open("./preserve/rnxrresult_nostbl{}.pkl".format(sys.argv[1]) ,mode='wb') as f:
    pickle.dump(array(exc_s.s),f)
    pickle.dump(weighte_init,f)
    pickle.dump(len(M.i),f)
    pickle.dump(target,f)
    pickle.dump(Mdopa.prefer,f)
    pickle.dump(Mdopa.crr,f)





