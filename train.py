import argparse

import torch
import cv2
import numpy as np
import os

from model import Generator, Discriminator, VGG19

torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
    
def load_image(image_path, x32=False):
    img = cv2.imread(image_path).astype(np.float32)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]

    if x32: # resize image to multiple of 32s
        def to_32s(x):
            return 256 if x < 256 else x - x%32
        img = cv2.resize(img, (to_32s(w), to_32s(h)))

    img = torch.from_numpy(img)
    img = img/127.5 - 1.0
    return img

def train(args):
    device = args.device
    
    netG = Generator()
    netD = Discriminator()
    vgg19 = VGG19(feature_mode=True)

    optimizerG = torch.optim.Adam(G_net.parameters(), lr=args.lrG)
    optimizerD = torch.optim.Adam(D_net.parameters(), lr=args.lrD)

    train_loader = utils.data_load(os.path.join('data', args.input_dir), 'train', args.batch_size, shuffle=True, drop_last=True)

    for epoch in range(args.train_epoch):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            # train generator G
            optimizerG.zero_grad()
            fake = netG(x, args.upsample_align).squeeze(0).permute(1, 2, 0)

            discriminator_loss = (netD(fake) - 1) ** 2
            content_loss = torch.nn.functional.l1_loss(vgg19(fake), vgg19(x))
            lossG = 300 * discriminator_loss + 1.5 * content_loss

            lossG.backward()
            optimizerG.step()

            # train discriminator D
            optimizerD.zero_grad()
            lossDFake = torch.nn.functional.binary_cross_entropy(netD(fake), 0)
            lossDReal = torch.nn.functional.binary_cross_entropy(netD(y), 1)
            lossD = lossDFake + lossDReal

            lossD.backward()
            optimizerD.step()

    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='./dataset')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--device', type=str,default='cuda:0')
    parser.add_argument('--train_epoch', type=int, default=100)
    parser.add_argument('--lrD', type=float, default=4e-5, help='learning rate, default=0.0002')
    parser.add_argument('--lrG', type=float, default=2e-5, help='learning rate, default=0.0002')
    parser.add_argument('--upsample_align', type=bool, default=False)

    args = parser.parse_args()
    
    train(args)
    
