import torch
import torch.nn as nn
import torch.nn.functional as F

def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

# https://pytorch.org/tutorials/advanced/neural_style_tutorial.html
def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.reshape(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)

# https://kornia.readthedocs.io/en/latest/_modules/kornia/color/yuv.html
def rgb_to_yuv(image: torch.Tensor) -> torch.Tensor:
    r"""Convert an RGB image to YUV.

    The image data is assumed to be in the range of (0, 1).

    Args:
        image (torch.Tensor): RGB Image to be converted to YUV with shape :math:`(*, 3, H, W)`.

    Returns:
        torch.Tensor: YUV version of the image with shape :math:`(*, 3, H, W)`.

    Example:
        >>> input = torch.rand(2, 3, 4, 5)
        >>> output = rgb_to_yuv(input)  # 2x3x4x5
    """
    if not isinstance(image, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(image)))

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError("Input size must have a shape of (*, 3, H, W). Got {}"
                         .format(image.shape))

    r: torch.Tensor = image[..., 0, :, :]
    g: torch.Tensor = image[..., 1, :, :]
    b: torch.Tensor = image[..., 2, :, :]

    y: torch.Tensor = 0.299 * r + 0.587 * g + 0.114 * b
    u: torch.Tensor = -0.147 * r - 0.289 * g + 0.436 * b
    v: torch.Tensor = 0.615 * r - 0.515 * g - 0.100 * b

    return y, u, v

def con_sty_loss(vgg, real, generated, grayscale):
    vgg_p = vgg((real + 1) / 2)
    vgg_G_p = vgg((generated + 1) / 2)
    vgg_x = vgg((grayscale + 1) / 2)
    return F.l1_loss(vgg_G_p, vgg_p.detach()), F.l1_loss(gram_matrix(vgg_G_p), gram_matrix(vgg_x.detach()))

def color_loss(real, generated):
    Y_G_p, U_G_p, V_G_p = rgb_to_yuv((generated + 1) / 2)
    Y_p, U_p, V_p = rgb_to_yuv((real + 1) / 2)
    return F.l1_loss(Y_G_p, Y_p) + F.smooth_l1_loss(U_G_p, U_p) + F.smooth_l1_loss(V_G_p, V_p)

# https://discuss.pytorch.org/t/implement-total-variation-loss-in-pytorch/55574/2
def total_variation_loss(img):
     bs_img, c_img, h_img, w_img = img.size()
     tv_h = torch.pow(img[:,:,1:,:]-img[:,:,:-1,:], 2).sum()
     tv_w = torch.pow(img[:,:,:,1:]-img[:,:,:,:-1], 2).sum()
     return (tv_h+tv_w)/(bs_img*c_img*h_img*w_img)
