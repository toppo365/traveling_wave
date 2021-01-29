import os
import time
from multiprocessing import Pool
import os

os.makedirs("./preserve/", exist_ok=True)
name = ["_nobl","_nostbl","_nowvbl","_org"]

timestamp = []

nprocess = 15
nn = 50

print(name)
def command(cmd):
    os.system(cmd)
while True:
    cmd = []
    for suf in name:
        dr = os.listdir("./preserve/")
        for i in range(nn):
            flag = 0
            st = "result{}{:03}".format(suf,i)
            check = [1 for dnm in dr if st in dnm]
            if len(check)>0:
                continue
            cmd.append('python ff_xor{}.py {:03}'.format(suf,i))
    if len(cmd) == 0:
        break

    print(cmd[::nprocess])

    with Pool(processes=nprocess) as pool:
        pool.map(command, cmd)

checkcmd = str(nn)
for nm in name:
    checkcmd += " {}".format(nm)

