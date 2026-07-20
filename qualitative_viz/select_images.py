import os
import sys

import numpy as np
from PIL import Image

root = sys.argv[1] if len(sys.argv) > 1 else \
    r'D:\.mail\my\Code\s2_corr-main\data\VOCdevkit\VOC2012'
val = open(os.path.join(root, 'ImageSets', 'Segmentation', 'val.txt')).read().split()
# bicycle=2, bottle=5, chair=9, person=15, pottedplant=16, sofa=18
targets = {2, 5, 9, 15, 16, 18}
scored = []
for v in val:
    g = np.array(Image.open(os.path.join(root, 'SegmentationClass', v + '.png')))
    present = targets & set(int(x) for x in np.unique(g))
    if len(present) >= 2:
        scored.append((len(present), v, sorted(present)))
scored.sort(reverse=True)
for n, v, p in scored[:10]:
    print(n, v, p)
