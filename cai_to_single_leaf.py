import numpy as np
import open3d as o3d
import os
import stat
import glob
from tqdm import tqdm
import argparse


def main(in_path: str, out_path: str):
    files = glob.glob(os.path.join(in_path, '*.txt'))
    for file in tqdm(files):
        os.chmod(file, stat.S_IRWXU)
        data = np.loadtxt(file)
        data = data[data[:, -2] == 1]
        if len(data) == 0:
            continue

        inst_labels = data[:, -1]
        inst_max = int(inst_labels.max() + 1)

        j = 0
        file_name = os.path.basename(file)[:-4]
        # this_out_path = os.path.join(out_path, file_name)
        this_out_path = out_path
        os.makedirs(this_out_path, exist_ok=True)
        for i in range(inst_max):
            ind = np.where(i == inst_labels)[0]
            if ind.shape[0]:
                j += 1
                out_file_name = file_name + f"-{str(j)}.txt"
                np.savetxt(os.path.join(this_out_path, out_file_name), data[ind,:6], fmt="%f %f %f %d %d %d")

if __name__ == "__main__":
    parser = argparse.ArgumentParser('')
    parser.add_argument('--in_path', type=str)
    parser.add_argument('--out_path', type=str)
    args = parser.parse_args()

    file_path = args.in_path
    out_path = args.out_path

    main(file_path, out_path)
