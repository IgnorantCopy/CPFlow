import os
import glob


path = "dataset/Ago/process"
filenames = glob.glob(os.path.join(path, '*.pt'))
for filename in filenames:
    if filename.split('.')[-2].endswith("-1"):
        # remove this file
        os.remove(filename)