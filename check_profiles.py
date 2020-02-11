
import json
import os
import h5py

def main():
    state   = filter(lambda name: ".workload" in name, os.listdir())
    with h5py.File(list(state)[0], "r") as f:
        for i in f.keys():
            if (i == "0"):
                continue
            if (f[i].shape[0]) < 64:
                exit(1)
    exit(0)

if __name__ == "__main__":
    main()
