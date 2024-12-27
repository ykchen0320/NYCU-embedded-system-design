import os
import glob

pic_dir = 'dataset/bob'
files = glob.glob(os.path.join(pic_dir, '*.png'))
for f in files:
	os.rename(f, f.split('_')[2])
