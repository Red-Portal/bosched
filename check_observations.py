
import json
import os

def main():
    state   = filter(lambda name: ".bostate" in name, os.listdir())
    with open(list(state)[0], "r") as jsonf:
        bostate = json.load(jsonf)
        for i in bostate["loops"]:
            if not("obs_x" in i) or len(i["obs_x"]) < 64:
                exit(1)
    exit(0)

if __name__ == "__main__":
    main()
