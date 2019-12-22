
import json
import os
import numpy as np

def main():
    state   = filter(lambda name: ".bostate" in name, os.listdir())
    with open(list(state)[0], "r") as jsonf:
        bostate = json.load(jsonf)
        for i in bostate["loops"]:
            if not("obs_x" in i):
                exit(1)
            else:
                if np.array(i["obs_x"]).flatten().shape[0] < 128:
                    exit(1)
    exit(0)

if __name__ == "__main__":
    main()
