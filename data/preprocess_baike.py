"""preprocess baike corpus
"""

import json
import os
from os.path import join
from tqdm import tqdm


def process(input_file, output_file):
    fout = open(output_file, "w")
    with open(input_file, "r") as f:
        for line in tqdm(f):
            jobj = json.loads(line)
            if jobj["summary"] is not None:
                text = jobj["summary"].replace("\n", "<n>") + "\n"
                if len(text) < 8:
                    pass
                else:
                    fout.write(text)
            for s in jobj["sections"]:
                text = s["content"].replace("\n", "<n>") + "\n"
                if len(text) < 8:
                    pass
                else:
                    fout.write(text)
    fout.close()


if __name__ == "__main__":
    input_file = "data/563w_baidubaike.json"
    output_file = "data/preprocessed/baike_train"
    process(input_file, output_file)