import argparse, time

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from model import Generator, Discriminator, VGG19, VideoGenerator, VideoDiscriminator
from data_mean import get_mean

from dataset import ImageDataset
import utils
from torch.utils.tensorboard import SummaryWriter

# torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def train(args):
    device = torch.device('cuda' if args.cuda else 'cpu')
    
    netG = VideoGenerator().to(device)
    netD = Discriminator().to(device)
    netVD = VideoDiscriminator().to(device)
    vgg19 = VGG19(init_weights='vgg19-dcbb9e9d.pth', feature_mode=True).to(device)
    vgg19.eval()

    optimizerG = torch.optim.Adam(netG.parameters(), lr=args.lrG)
    optimizerD = torch.optim.Adam(netD.parameters(), lr=args.lrD)
    optimizerVD = torch.optim.Adam(netVD.parameters(), lr=args.lrVD)

    data_mean = get_mean(f'{args.dataset}/anime')
    anime_style = ImageDataset('dataset' + '/anime/style', real=False, data_mean=data_mean)
    anime_smooth = ImageDataset('dataset' + '/anime/smooth', real=False, data_mean=data_mean)
    real = ImageDataset('dataset' + '/real')

    anime_style_loader = DataLoader(anime_style, args.batch_size, shuffle=True, drop_last=True)
    anime_smooth_loader = DataLoader(anime_smooth, args.batch_size, shuffle=True, drop_last=True)
    real_loader = DataLoader(real, args.batch_size, shuffle=True, drop_last=True)

    if args.use_writer:
        writer = SummaryWriter(comment=f'Exp_{args.exp_name}')

    # netG.load_state_dict(torch.load('./pytorch_generator_Paprika.pt'))
    # netD.load_state_dict(torch.load('./checkpoints/Discriminator/netD_100.pt'))

    netG.load_state_dict(torch.load('./checkpoints/netG_pretrained_5.pt'))

    pretrain_iter = 0
    for epoch in range(args.pre_train_epoch):
        recon_losses = []
        epoch_start_time = time.time()
        for x_prev, _, x, _ in real_loader:
            if (pretrain_iter+1) % 100 == 0:
                print('pretrain_iter', pretrain_iter+1, torch.mean(torch.FloatTensor(recon_losses)))

            x_prev = x_prev.to(device)
            x = x.to(device)

            x_prev = x_prev.permute(0, 3, 1, 2)
            x = x.permute(0, 3, 1, 2)

            # pre-train generator G
            optimizerG.zero_grad()

            vgg_p = vgg19((x + 1) / 2)
            pre_G_p = netG(utils.cat_images(x_prev, x_prev), args.upsample_align).detach()
            G_p = netG(utils.cat_images(x, pre_G_p), args.upsample_align)
            vgg_G_p = vgg19((G_p + 1) / 2)

            lossG = F.l1_loss(vgg_G_p, vgg_p.detach())
            recon_losses.append(lossG.item())

            lossG.backward()
            optimizerG.step()

            pretrain_iter += 1
        
        per_epoch_time = time.time() - epoch_start_time
        recon_losses = torch.mean(torch.FloatTensor(recon_losses))
        print('[%d/%d] - time: %.2fs, Recon loss: %.3f' % ((epoch + 1), args.pre_train_epoch, per_epoch_time, recon_losses))

        torch.save(netG.state_dict(), f'checkpoints/netG_pretrained_{epoch + 1}.pt')

        if args.use_writer:
            writer.add_scalar('Loss/pretrain', recon_losses, epoch + 1)

    iters = 0
    print('===============start training==================')
    for epoch in range(args.train_epoch):
        d_losses = []
        vd_losses = []
        g_losses = []

        epoch_start_time = time.time()

        anime_style_loader_iter = iter(anime_style_loader)
        anime_smooth_loader_iter = iter(anime_smooth_loader)
        real_loader_iter = iter(real_loader)

        while True:
            try:
                anime_style_data_prev, _, anime_style_data, anime_style_grayscale_data = next(anime_style_loader_iter)
                _, _, _, anime_smooth_grayscale_data = next(anime_smooth_loader_iter)
                real_data_prev, _, real_data, _ = next(real_loader_iter)
            except StopIteration:
                break

            if (iters+1) % 100 == 0:
                print('iter', iters+1, 'd_loss', torch.mean(torch.FloatTensor(d_losses)), 'vd_loss', torch.mean(torch.FloatTensor(vd_losses)), 'g_loss', torch.mean(torch.FloatTensor(g_losses)))

            anime_style_data_prev = anime_style_data_prev.to(device)
            anime_style_data = anime_style_data.to(device)
            anime_style_grayscale_data = anime_style_grayscale_data.to(device)
            anime_smooth_grayscale_data = anime_smooth_grayscale_data.to(device)
            real_data_prev = real_data_prev.to(device)
            real_data = real_data.to(device)

            anime_style_data_prev = anime_style_data_prev.permute(0, 3, 1, 2)
            anime_style_data = anime_style_data.permute(0, 3, 1, 2)
            anime_style_grayscale_data = anime_style_grayscale_data.permute(0, 3, 1, 2)
            anime_smooth_grayscale_data = anime_smooth_grayscale_data.permute(0, 3, 1, 2)
            real_data_prev = real_data_prev.permute(0, 3, 1, 2)
            real_data = real_data.permute(0, 3, 1, 2)

            # train generator G
            optimizerG.zero_grad()
            pre_generated = netG(utils.cat_images(real_data_prev, real_data_prev), args.upsample_align).detach()
            generated = netG(utils.cat_images(real_data, pre_generated), args.upsample_align)
            generated_logits = netD(generated)
            generated_video_logits = netVD(utils.cat_images(generated, pre_generated))

            adv_loss = torch.mean((generated_logits - 1) ** 2)
            video_adv_loss = torch.mean((generated_video_logits - 1) ** 2)
            content_loss, style_loss = utils.con_sty_loss(vgg19, real_data, generated, anime_style_grayscale_data)
            color_loss = utils.color_loss(real_data, generated)
            tv_loss = utils.total_variation_loss(generated)
            consistency_loss = F.l1_loss(generated, pre_generated)
            lossG = adv_loss + video_adv_loss + 1.2 * content_loss + 3 * style_loss + 9 * color_loss + tv_loss + 0.5 * consistency_loss
            g_losses.append(lossG.item())

            lossG.backward()
            optimizerG.step()

            # train discriminator D
            optimizerD.zero_grad()
            anime_logits = netD(anime_style_data)
            pre_generated = netG(utils.cat_images(real_data_prev, real_data_prev), args.upsample_align).detach()
            generated = netG(utils.cat_images(real_data, pre_generated), args.upsample_align)
            generated_logits = netD(generated)
            anime_gray_logits = netD(anime_style_grayscale_data)
            anime_smooth_logits = netD(anime_smooth_grayscale_data)

            lossD = (1.7 * torch.mean((anime_logits - 1) ** 2) + 1.7 * torch.mean(generated_logits ** 2) + 1.7 * torch.mean(anime_gray_logits ** 2) + torch.mean(anime_smooth_logits ** 2))
            d_losses.append(lossD.item())

            lossD.backward()
            optimizerD.step()

            # train discriminator VD
            optimizerVD.zero_grad()
            anime_logits = netVD(utils.cat_images(anime_style_data, anime_style_data_prev))
            pre_generated = netG(utils.cat_images(real_data_prev, real_data_prev), args.upsample_align).detach()
            generated = netG(utils.cat_images(real_data, pre_generated), args.upsample_align)
            generated_logits = netVD(utils.cat_images(generated, pre_generated))

            lossVD = 1.7 * torch.mean((anime_logits - 1) ** 2) + 1.7 * torch.mean(generated_logits ** 2)
            vd_losses.append(lossVD.item())

            lossVD.backward()
            optimizerVD.step()

            iters += 1

        per_epoch_time = time.time() - epoch_start_time
        d_losses = torch.mean(torch.FloatTensor(d_losses))
        vd_losses = torch.mean(torch.FloatTensor(vd_losses))
        g_losses = torch.mean(torch.FloatTensor(g_losses))
        print('[%d/%d] - time: %.2fs, d_loss: %.3f, vd_loss: %.3f, g_loss: %.3f' % ((epoch + 1), args.train_epoch, per_epoch_time, d_losses, vd_losses, g_losses))
        torch.save(netG.state_dict(), f'checkpoints/netG_{epoch + 1}.pt')
        torch.save(netD.state_dict(), f'checkpoints/netD_{epoch + 1}.pt')
        torch.save(netD.state_dict(), f'checkpoints/netVD_{epoch + 1}.pt')

        if args.use_writer:
            writer.add_scalar('Loss/d_loss', d_losses, epoch + 1)
            writer.add_scalar('Loss/vd_loss', vd_losses, epoch + 1)
            writer.add_scalar('Loss/g_loss', g_losses, epoch + 1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='./dataset')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--train_epoch', type=int, default=100)
    parser.add_argument('--pre_train_epoch', type=int, default=5)
    parser.add_argument('--lrD', type=float, default=4e-5, help='learning rate, default=0.0004')
    parser.add_argument('--lrVD', type=float, default=4e-5, help='learning rate, default=0.0004')
    parser.add_argument('--lrG', type=float, default=2e-5, help='learning rate, default=0.0002')
    parser.add_argument('--upsample_align', type=bool, default=False)
    parser.add_argument('--use_writer', action='store_true', help='use tensorboard')
    parser.add_argument('--exp_name', type=str, default='debug', help='Name of the experiment')

    args = parser.parse_args()
    
    train(args)
    