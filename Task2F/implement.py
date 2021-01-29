import os
import time
from multiprocessing import Pool
import os

os.makedirs("./preserve/", exist_ok=True)

name = ["_nobl","_nostbl","_nowvbl","_org"]

timestamp = []
print(name)

nprocess = 10
nnum = 50

def command(cmd):
    os.system(cmd)

while True:
    dr = os.listdir("./preserve/")
    cmd = []
    for suf in name:
        t1 = time.time()
        for i in range(nnum):
            st = "{}{:03}".format(suf,i)
            check = [1 for dnm in dr if st in dnm]
            if len(check)>1:
                continue
            cmd.append('python pth11{}.py {:03}'.format(suf,i))
    if len(cmd) == 0:
        break
    print(len(cmd))
    print(cmd[::nprocess])

    with Pool(processes=nprocess) as pool:
        pool.map(command, cmd)