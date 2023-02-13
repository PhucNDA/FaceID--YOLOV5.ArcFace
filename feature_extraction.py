import argparse

import cv2
import numpy as np
import torch
import os
from pathlib import Path
from iresnet import iresnet100

@torch.no_grad()
def inference(net, img):
    if img is None:
        img = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.uint8)
    else:
        img = cv2.imread(img)
        img = cv2.resize(img, (112, 112))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).float()
    img.div_(255).sub_(0.5).div_(0.5)
    feat = net(img).numpy()
    return feat

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch ArcFace Training')
    parser.add_argument('--network', type=str, default='r100', help='backbone network')
    parser.add_argument('--weight', type=str, default='')
    parser.add_argument('--path_database', type=Path)
    args = parser.parse_args()
    check=os.path.exists('database_tensor')
    if check==False:
        os.mkdir('database_tensor')
    net = iresnet100(False)
    net.load_state_dict(torch.load(args.weight))
    net.eval()
    path=args.path_database
    import os
    img=os.listdir(path)
    for im in img:
        np.save('database_tensor/'+im.replace('.png','')+'.npy',inference(net, str(path)+'/'+im))
        print(im)