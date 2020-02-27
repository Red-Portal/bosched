
import json
import os
import sys
import numpy as np

def read_state(filename):
    with open(filename, "r") as jsonf:
        return json.load(jsonf)
    return

def main(argv):
    state = filter(lambda name: ".bostate" in name, os.listdir())
    if len(argv) > 0:
        state = filter(lambda name: argv[1] in name, state)
    state = list(state)
    print("Merging ", len(state), " BO state files")

    base    = read_state(state[0])
    baseids = [l["id"] for l in base["loops"]]
    datacnt = np.ones(len(baseids))

    for fname_i in state[1:]:
        candidate = read_state(fname_i)
        for loop in candidate["loops"]:
            loopid  = loop["id"]
            loopidx = baseids.index(loopid)
            sum_arr = np.array(base["loops"][loopidx]["hist_y"])
            add_arr = np.array(loop["hist_y"])

            if(sum_arr.shape != add_arr.shape):
                print("Different shape: ", sum_arr.shape, " ", add_arr.shape)
                exit(1)

            base["loops"][loopidx]["hist_y"] = (sum_arr + add_arr).tolist()
            datacnt[loopidx] += 1

    for loop in base["loops"]:
        loopid  = loop["id"]
        loopidx = baseids.index(loopid)
        data    = np.array(base["loops"][loopidx]["hist_y"])
        data   /= datacnt[loopidx]
        base["loops"][loopidx]["hist_y"] = data.tolist()

    print("loop merge counts = ", datacnt)
    print("baseids = ", baseids)

    for fname_i in state:
        with open(fname_i, "w") as jsonf:
            j = json.dumps(base, indent=2)
            jsonf.write(j)

    exit(0)

if __name__ == "__main__":
    main(sys.argv)
