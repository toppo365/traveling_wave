import os
import time
from multiprocessing import Pool
import os

os.makedirs("./preserve/", exist_ok=True)

name = ["_nobl","_nowvbl","_nostbl","_org"]

nn = 50
nprocess = 10

timestamp = []
print(name)

def command(cmd):
    os.system(cmd)
while True:
    dr = os.listdir("./preserve/")
    cmd = []
    for suf in name:
        t1 = time.time()
        for i in range(nn):
            st = "{}{:03}".format(suf,i)
            check = [1 for dnm in dr if st in dnm]
            if len(check)>0:
                continue
            cmd.append('python rec-1to4{}.py {:03}'.format(suf,i))
    if len(cmd) == 0:
        break
    print("{} jobs in tatal".format(len(cmd)))
    print(cmd[::nprocess])
    with Pool(processes=nprocess) as pool:
        pool.map(command, cmd)



