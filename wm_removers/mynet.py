import torch
import torch.nn as nn
import torch.nn.functional as F
from scripts.models.blocks import SEBlock
from scripts.models.rasc import *
from scripts.models.unet import UnetGenerator, MinimalUnetV2


def weight_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)


def reset_params(model):
    for i, m in enumerate(model.modules()):
        weight_init(m)


def conv3x3(in_channels, out_channels, stride=1,
            padding=1, bias=True, groups=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        bias=bias,
        groups=groups)


def up_conv2x2(in_channels, out_channels, transpose=True):
    if transpose:
        return nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2)
    else:
        return nn.Sequential(
            nn.Upsample(mode='bilinear', scale_factor=2),
            conv1x1(in_channels, out_channels))


def conv1x1(in_channels, out_channels, groups=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=1,
        groups=groups,
        stride=1)


class UpCoXvD(nn.Module):

    def __init__(self, in_channels, out_channels, blocks, residual=True, norm=nn.BatchNorm2d, act=F.relu,
                 batch_norm=True, transpose=True, concat=True, use_att=False):
        super(UpCoXvD, self).__init__()
        self.concat = concat
        self.residual = residual
        self.batch_norm = batch_norm
        self.bn = None
        self.conv2 = []
        self.use_att = use_att
        self.up_conv = up_conv2x2(in_channels, out_channels, transpose=transpose)
        self.norm0 = norm(out_channels)

        if self.use_att:
            self.s2am = RASC(2 * out_channels)
        else:
            self.s2am = None

        if self.concat:
            self.conv1 = conv3x3(2 * out_channels, out_channels)
            self.norm1 = norm(out_channels, out_channels)
        else:
            self.conv1 = conv3x3(out_channels, out_channels)
            self.norm1 = norm(out_channels, out_channels)

        for _ in range(blocks):
            self.conv2.append(conv3x3(out_channels, out_channels))
        if self.batch_norm:
            self.bn = []
            for _ in range(blocks):
                self.bn.append(norm(out_channels))
            self.bn = nn.ModuleList(self.bn)
        self.conv2 = nn.ModuleList(self.conv2)
        self.act = act

    def forward(self, from_up, from_down, mask=None, se=None):
        from_up = self.act(self.norm0(self.up_conv(from_up)))
        if self.concat:
            x1 = torch.cat((from_up, from_down), 1)
        else:
            if from_down is not None:
                x1 = from_up + from_down
            else:
                x1 = from_up

        if self.use_att:
            x1 = self.s2am(x1, mask)

        x1 = self.act(self.norm1(self.conv1(x1)))
        x2 = None
        for idx, conv in enumerate(self.conv2):
            x2 = conv(x1)
            if self.batch_norm:
                x2 = self.bn[idx](x2)

            if (se is not None) and (idx == len(self.conv2) - 1):  # last
                x2 = se(x2)

            if self.residual:
                x2 = x2 + x1
            x2 = self.act(x2)
            x1 = x2
        return x2


class DownCoXvD(nn.Module):

    def __init__(self, in_channels, out_channels, blocks, pooling=True, norm=nn.BatchNorm2d, act=F.relu, residual=True,
                 batch_norm=True):
        super(DownCoXvD, self).__init__()
        self.pooling = pooling
        self.residual = residual
        self.batch_norm = batch_norm
        self.bn = None
        self.pool = None
        self.conv1 = conv3x3(in_channels, out_channels)
        self.norm1 = norm(out_channels)

        self.conv2 = []
        for _ in range(blocks):
            self.conv2.append(conv3x3(out_channels, out_channels))
        if self.batch_norm:
            self.bn = []
            for _ in range(blocks):
                self.bn.append(norm(out_channels))
            self.bn = nn.ModuleList(self.bn)
        if self.pooling:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.ModuleList(self.conv2)
        self.act = act

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        x1 = self.act(self.norm1(self.conv1(x)))
        x2 = None
        for idx, conv in enumerate(self.conv2):
            x2 = conv(x1)
            if self.batch_norm:
                x2 = self.bn[idx](x2)
            if self.residual:
                x2 = x2 + x1
            x2 = self.act(x2)
            x1 = x2
        before_pool = x2
        if self.pooling:
            x2 = self.pool(x2)
        return x2, before_pool

class UnetDecoder(nn.Module):
    def __init__(self, in_channels=512, out_channels=3, norm=nn.BatchNorm2d, act=F.relu, depth=5, blocks=1,
                 residual=True, batch_norm=True,
                 transpose=True, concat=True, is_final=True, use_att=False):
        super(UnetDecoder, self).__init__()
        self.conv_final = None
        self.up_convs = []
        self.atts = []
        self.use_att = use_att

        outs = in_channels
        for i in range(depth - 1):  # depth = 1
            ins = outs
            outs = ins // 2
            # 512,256
            # 256,128
            # 128,64
            # 64,32
            up_conv = UpCoXvD(ins, outs, blocks, residual=residual, batch_norm=batch_norm, transpose=transpose,
                              concat=concat, norm=norm, act=act)
            if self.use_att:
                self.atts.append(SEBlock(outs))

            self.up_convs.append(up_conv)

        if is_final:
            self.conv_final = conv1x1(outs, out_channels)
        else:
            up_conv = UpCoXvD(outs, out_channels, blocks, residual=residual, batch_norm=batch_norm, transpose=transpose,
                              concat=concat, norm=norm, act=act)
            if self.use_att:
                self.atts.append(SEBlock(out_channels))

            self.up_convs.append(up_conv)
        self.up_convs = nn.ModuleList(self.up_convs)
        self.atts = nn.ModuleList(self.atts)

        reset_params(self)

    def __call__(self, x, encoder_outs=None):
        return self.forward(x, encoder_outs)

    def forward(self, x, encoder_outs=None):
        for i, up_conv in enumerate(self.up_convs):
            before_pool = None
            if encoder_outs is not None:
                before_pool = encoder_outs[-(i + 2)]
            x = up_conv(x, before_pool)
            if self.use_att:
                x = self.atts[i](x)

        if self.conv_final is not None:
            x = self.conv_final(x)
        return x
class UnetDecoderNotShared(nn.Module):
    def __init__(self, in_channels=512, out_channels=3, norm=nn.BatchNorm2d, act=F.relu, depth=5, blocks=1,
                 residual=True, batch_norm=True,
                 transpose=True, concat=True, is_final=True, use_att=True):
        super(UnetDecoderNotShared, self).__init__()
        self.conv_final = None
        self.up_convs = []
        self.atts = []
        self.use_att = use_att

        outs = in_channels
        for i in range(depth - 1):  # depth = 1
            ins = outs
            outs = ins // 2
            # 512,256
            # 256,128
            # 128,64
            # 64,32
            up_conv = UpCoXvD(ins, outs, blocks, residual=residual, batch_norm=batch_norm, transpose=transpose,
                              concat=concat, norm=norm, act=act)
            if self.use_att:
                self.atts.append(SEBlock(outs))

            self.up_convs.append(up_conv)

        if is_final:
            self.conv_final = conv1x1(outs, out_channels)
        else:
            up_conv = UpCoXvD(outs, out_channels, blocks, residual=residual, batch_norm=batch_norm, transpose=transpose,
                              concat=concat, norm=norm, act=act)
            if self.use_att:
                self.atts.append(SEBlock(out_channels))

            self.up_convs.append(up_conv)
        self.up_convs = nn.ModuleList(self.up_convs)
        self.atts = nn.ModuleList(self.atts)

        reset_params(self)

    def __call__(self, x, encoder_outs=None):
        return self.forward(x, encoder_outs)

    def forward(self, x, encoder_outs=None):
        for i, up_conv in enumerate(self.up_convs):
            before_pool = None
            if encoder_outs is not None:
                before_pool = encoder_outs[-(i + 2)]
            x = up_conv(x, before_pool)
            if self.use_att:
                x = self.atts[i](x)

        if self.conv_final is not None:
            x = self.conv_final(x)
        return x


class UnetDecoderShared(nn.Module):
    def __init__(self, in_channels=512, out_channels=3, depth=5, blocks=1, residual=True, batch_norm=True,
                 transpose=True, concat=True, is_final=True, norm=nn.BatchNorm2d, act=F.relu):
        super(UnetDecoderShared, self).__init__()
        self.conv_final = None
        self.up_convs = []
        self.im_atts = []
        self.vm_atts = []
        # self.mask_atts = []

        outs = in_channels
        for i in range(depth - 1):  # depth = 5 [0,1,2,3]
            ins = outs
            outs = ins // 2
            # 512,256
            # 256,128
            # 128,64
            # 64,32
            up_conv = UpCoXvD(ins, outs, blocks, residual=residual, batch_norm=batch_norm, transpose=transpose,
                              concat=concat, norm=nn.BatchNorm2d, act=F.relu)
            self.up_convs.append(up_conv)
            self.im_atts.append(SEBlock(outs))
            self.vm_atts.append(SEBlock(outs))
            # self.mask_atts.append(SEBlock(outs))
        if is_final:
            self.conv_final = conv1x1(outs, out_channels)
        else:
            up_conv = UpCoXvD(outs, out_channels, blocks, residual=residual, batch_norm=batch_norm, transpose=transpose,
                              concat=concat, norm=nn.BatchNorm2d, act=F.relu)
            self.up_convs.append(up_conv)
            self.im_atts.append(SEBlock(out_channels))
            self.vm_atts.append(SEBlock(out_channels))
            # self.mask_atts.append(SEBlock(out_channels))

        self.up_convs = nn.ModuleList(self.up_convs)
        self.im_atts = nn.ModuleList(self.im_atts)
        self.vm_atts = nn.ModuleList(self.vm_atts)
        # self.mask_atts = nn.ModuleList(self.mask_atts)

        reset_params(self)

    def forward(self, input, encoder_outs=None):
        # im branch
        x = input
        for i, up_conv in enumerate(self.up_convs):
            before_pool = None
            if encoder_outs is not None:
                # 倒数元素取值
                index = -(i+2)
                if index != -6:
                    before_pool = encoder_outs[index]
            if index != -6:
                x = up_conv(x, before_pool, se=self.im_atts[i])
        # 最后一个卷积
        if self.conv_final is not None:
            x = self.conv_final(x)
        x_im = x
        # x = input
        # for i, up_conv in enumerate(self.up_convs):
        #     before_pool = None
        #     if encoder_outs is not None:
        #         before_pool = encoder_outs[-(i + 2)]
        #     x = up_conv(x, before_pool, se=self.mask_atts[i])
        # x_mask = x

        x = input
        for i, up_conv in enumerate(self.up_convs):
            before_pool = None
            if encoder_outs is not None:
                # 倒数元素取值
                index = -(i+2)
                if index != -6:
                    before_pool = encoder_outs[index]
            if index != -6:
                x = up_conv(x, before_pool, se=self.im_atts[i])
        if self.conv_final is not None:
            x = self.conv_final(x)
        x_vm = x

        return x_im, x_vm


class UnetEncoderD(nn.Module):

    def __init__(self, in_channels=3, depth=5, blocks=1, start_filters=32, residual=True, batch_norm=True,
                 norm=nn.BatchNorm2d, act=F.relu):
        super(UnetEncoderD, self).__init__()
        self.down_convs = []
        outs = None
        if type(blocks) is tuple:
            blocks = blocks[0]
        for i in range(depth):
            ins = in_channels if i == 0 else outs
            outs = start_filters * (2 ** i)
            pooling = True if i < depth - 1 else False
            down_conv = DownCoXvD(ins, outs, blocks, pooling=pooling, residual=residual, batch_norm=batch_norm,
                                  norm=nn.BatchNorm2d, act=F.relu)
            self.down_convs.append(down_conv)
        self.down_convs = nn.ModuleList(self.down_convs)
        reset_params(self)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        encoder_outs = []
        for d_conv in self.down_convs:
            x, before_pool = d_conv(x)
            encoder_outs.append(before_pool)
        return x, encoder_outs


class ResDown(nn.Module):
    def __init__(self, in_size, out_size, pooling=True, use_att=False):
        super(ResDown, self).__init__()
        self.model = DownCoXvD(in_size, out_size, 3, pooling=pooling)

    def forward(self, x):
        return self.model(x)


class ResUp(nn.Module):
    def __init__(self, in_size, out_size, use_att=False):
        super(ResUp, self).__init__()
        self.model = UpCoXvD(in_size, out_size, 3, use_att=use_att)

    def forward(self, x, skip_input, mask=None):
        return self.model(x, skip_input, mask)


class ResDownNew(nn.Module):
    def __init__(self, in_size, out_size, pooling=True, use_att=False):
        super(ResDownNew, self).__init__()
        self.model = DownCoXvD(in_size, out_size, 3, pooling=pooling, norm=nn.InstanceNorm2d, act=F.leaky_relu)

    def forward(self, x):
        return self.model(x)


class ResUpNew(nn.Module):
    def __init__(self, in_size, out_size, use_att=False):
        super(ResUpNew, self).__init__()
        self.model = UpCoXvD(in_size, out_size, 3, use_att=use_att, norm=nn.InstanceNorm2d)

    def forward(self, x, skip_input, mask=None):
        return self.model(x, skip_input, mask)



class VMSingleS2AM(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, down=ResDown, up=ResUp, ngf=32):
        super(VMSingleS2AM, self).__init__()

        self.down1 = down(in_channels, ngf)
        self.down2 = down(ngf, ngf * 2)
        self.down3 = down(ngf * 2, ngf * 4)
        self.down4 = down(ngf * 4, ngf * 8)
        self.down5 = down(ngf * 8, ngf * 16, pooling=False)

        self.up1 = up(ngf * 16, ngf * 8)
        self.up2 = up(ngf * 8, ngf * 4)
        self.s2am2 = RASC(ngf * 4)

        self.up3 = up(ngf * 4, ngf * 2)
        self.s2am3 = RASC(ngf * 2)

        self.up4 = up(ngf * 2, ngf * 1)
        self.s2am4 = RASC(ngf)

        self.im = nn.Conv2d(ngf, 3, 1)

    def forward(self, input):
        img, mask = input[:, 0:3, :, :], input[:, 3:4, :, :]
        # U-Net generator with skip connections from encoder to decoder
        x, d1 = self.down1(input)  # 128,256
        x, d2 = self.down2(x)  # 64,128
        x, d3 = self.down3(x)  # 32,64
        x, d4 = self.down4(x)  # 16,32
        x, _ = self.down5(x)  # 8,16

        x = self.up1(x, d4)  # 16
        x = self.up2(x, d3)  # 32
        x = self.s2am2(x, mask)

        x = self.up3(x, d2)  # 64
        x = self.s2am3(x, mask)

        x = self.up4(x, d1)  # 128
        x = self.s2am4(x, mask)
        im = self.im(x)
        return im

class ResUNet(nn.Module):
# 标准的UNet
    def __init__(self, in_channels=3, depth=5, shared_depth=0, use_vm_decoder=False, blocks=1,
                 out_channels_image=3, out_channels_mask=1, start_filters=32, residual=True, batch_norm=True,
                 transpose=True, concat=True, transfer_data=True, long_skip=False, s2am='unet', use_coarser=True,
                 no_stage2=False, num=0):   # num的意思是层数0，1，2，3，对应2，3，4，5层U-Net
        super(ResUNet, self).__init__()
        self.transfer_data = transfer_data
        self.shared = shared_depth
        self.optimizer_encoder, self.optimizer_image, self.optimizer_vm = None, None, None
        self.optimizer_mask, self.optimizer_shared = None, None
        if type(blocks) is not tuple:
            blocks = (blocks, blocks, blocks, blocks)
            # encoder block, mask block, shared block
        if not transfer_data:
            concat = False
        # 编码器
        tdepth = num + 2
        self.encoder = UnetEncoderD(in_channels=in_channels, depth=tdepth, blocks=blocks[0],
                                    start_filters=start_filters, residual=residual, batch_norm=batch_norm,
                                    norm=nn.InstanceNorm2d, act=F.leaky_relu)
        tin_channels = start_filters * 2 ** (depth - 4 + num)
        self.image_decoder = UnetDecoder(in_channels=tin_channels,
                                         out_channels=out_channels_image, depth=num + 2,      # 因为上采样层有 i + 2 - 1个
                                         blocks=blocks[1], residual=residual, batch_norm=batch_norm,
                                         transpose=transpose, concat=concat, norm=nn.InstanceNorm2d)
        self.mask_decoder = UnetDecoder(in_channels=tin_channels,
                                         out_channels=out_channels_mask, depth=num + 2,      # 因为上采样层有 i + 2 - 1个
                                         blocks=blocks[2], residual=residual, batch_norm=batch_norm,
                                         transpose=transpose, concat=concat, norm=nn.InstanceNorm2d)
        self.wm_decoder = UnetDecoder(in_channels=tin_channels,
                                         out_channels=out_channels_image, depth=num + 2,      # 因为上采样层有 i + 2 - 1个
                                         blocks=blocks[3], residual=residual, batch_norm=batch_norm,
                                         transpose=transpose, concat=concat, norm=nn.InstanceNorm2d)
        self.use_coarser = use_coarser
        self.long_skip = long_skip
        self.no_stage2 = no_stage2
        self._forward = self.unshared_forward
        if self.shared != 0:
            self._forward = self.shared_forward
            self.shared_decoder = UnetDecoderShared(in_channels=start_filters * 2 ** (depth - 1),
                                                  out_channels=out_channels_image,
                                                  depth=shared_depth, blocks=blocks[4], residual=residual,
                                                  batch_norm=batch_norm, transpose=transpose, concat=concat,
                                                  is_final=True, norm=nn.InstanceNorm2d)

        if s2am == 'vms2am':
            self.s2am = VMSingleS2AM(4, down=ResDownNew, up=ResUpNew)

    def set_optimizers(self):
        self.optimizer_encoder =  torch.optim.Adam(self.encoder.parameters(), lr=0.001)
        self.optimizer_image =  torch.optim.Adam(self.image_decoder.parameters(), lr=0.001)
        self.optimizer_mask = torch.optim.Adam(self.mask_decoder.parameters(), lr=0.001)
        self.optimizer_wm = torch.optim.Adam(self.wm_decoder.parameters(), lr=0.001)
        self.optimizer_s2am = torch.optim.Adam(self.s2am.parameters(), lr=0.001)
        if self.shared != 0:
            self.optimizer_shared = torch.optim.Adam(self.shared_decoder.parameters(), lr=0.001)   # 共享参数

    def zero_grad_all(self):
        self.optimizer_encoder.zero_grad()
        self.optimizer_image.zero_grad()
        self.optimizer_mask.zero_grad()
        self.optimizer_wm.zero_grad()
        self.optimizer_s2am.zero_grad()
        if self.shared != 0:
            self.optimizer_shared.zero_grad()

    def step_all(self):
        self.optimizer_encoder.step()
        self.optimizer_image.step()
        self.optimizer_mask.step()
        self.optimizer_wm.step()
        self.optimizer_s2am.step()
        if self.shared != 0:
            self.optimizer_shared.step()

    def __call__(self, synthesized):
        return self._forward(synthesized)

    def forward(self, synthesized):
        return self._forward(synthesized)

    def unshared_forward(self, synthesized):  # 当shared_depth = 0
        image_code, before_pool = self.encoder(synthesized)
        if not self.transfer_data:
            before_pool = None
        reconstructed_image = torch.tanh(self.image_decoder(image_code, before_pool))
        reconstructed_mask = torch.sigmoid(self.mask_decoder(image_code, before_pool))
        reconstructed_wm = torch.tanh(self.wm_decoder(image_code, before_pool))
        return reconstructed_image, reconstructed_mask, reconstructed_wm

    def shared_forward(self, synthesized):
        # 这里是shared_depth不为0时的代码
        index = 0
        refines = []
        reconstructed_images = []
        reconstructed_masks = []
        reconstructed_vms = []
        for encoder in self.encoders:
            encoder = encoder.to(synthesized.device)
        for decoder in self.image_decoders:
            decoder = decoder.to(synthesized.device)
        for decoder in self.image_decoders:
            decoder = decoder.to(synthesized.device)

        for encoder in self.encoders:
            encoder = encoder.to(synthesized.device)
            print(synthesized.device)
            print(encoder.device)
            image_code, before_pool = encoder(synthesized)
            reconstructed_image = torch.tanh(self.image_decoders[index](image_code, before_pool))
            reconstructed_mask = torch.tanh(self.mask_decoders[index](image_code, before_pool))
            reconstructed_wm = torch.tanh(self.wm_decoders[index](image_code, before_pool))
            if self.long_skip:
                reconstructed_image = reconstructed_image + synthesized
                reconstructed_vm = reconstructed_vm + synthesized
            coarser = reconstructed_image * reconstructed_mask + (1 - reconstructed_mask) * synthesized
            if self.use_coarser:
                refine = torch.tanh(self.s2am(torch.cat([coarser, reconstructed_mask], dim=1))) + coarser
            elif self.no_stage2:
                refine = torch.tanh(self.s2am(torch.cat([coarser, reconstructed_mask], dim=1)))
            else:
                refine = torch.tanh(self.s2am(torch.cat([coarser, reconstructed_mask], dim=1))) + synthesized
            refines.append(refine)
            reconstructed_images.append(reconstructed_image)
            reconstructed_masks.append(reconstructed_mask)
            reconstructed_vms.append(reconstructed_wm)
        res = np.zeros((4, 3, 256, 256))
        res = torch.from_numpy(res)
        res_image = np.zeros((4, 3, 256, 256))
        res_image = torch.from_numpy(res_image)
        res_mask = np.zeros((4, 3, 256, 256))
        res_mask = torch.from_numpy(res_mask)
        res_wm = np.zeros((4, 3, 256, 256))
        res_wm = torch.from_numpy(res_wm)
        for i in range(4):
            res += refines[i]
            res_image += reconstructed_images[i]
            res_mask += reconstructed_masks[i]
            res_wm += reconstructed_vms[i]

        return [res, res_image], res_mask, res_wm
class MyNet(nn.Module):

    def __init__(self, in_channels=3, depth=5, shared_depth=0, use_vm_decoder=False, blocks=1,
                 out_channels_image=3, out_channels_mask=1, start_filters=32, residual=True, batch_norm=True,
                 transpose=True, concat=True, transfer_data=True, long_skip=False, s2am='unet', use_coarser=True,
                 no_stage2=False):
        super(MyNet, self).__init__()
        self.transfer_data = transfer_data
        self.shared = shared_depth
        self.optimizer_encoder, self.optimizer_image, self.optimizer_vm = None, None, None
        self.optimizer_mask, self.optimizer_shared = None, None
        if type(blocks) is not tuple:
            blocks = (blocks, blocks, blocks, blocks, blocks, blocks, blocks, blocks, blocks, blocks, blocks, blocks, blocks, blocks, blocks, blocks)
            # encoder block, mask block, shared block
        if not transfer_data:
            concat = False
        # 编码器
        self.encoders =[]
        for i in range(4):
            tdepth = i + 2
            encoder = UnetEncoderD(in_channels=in_channels, depth=tdepth, blocks=blocks[i],
                                    start_filters=start_filters, residual=residual, batch_norm=batch_norm,
                                    norm=nn.InstanceNorm2d, act=F.leaky_relu)
            self.encoders.append(encoder)
        self.image_decoders = []
        for i in range(4):
            tin_channels = start_filters * 2 ** (depth - 4 + i)
            decoder = UnetDecoderNotShared(in_channels=tin_channels,
                                         out_channels=out_channels_image, depth=i + 2,      # 因为上采样层有 i + 2 - 1个
                                         blocks=blocks[i + 4], residual=residual, batch_norm=batch_norm,
                                         transpose=transpose, concat=concat, norm=nn.InstanceNorm2d)
            self.image_decoders.append(decoder)

        self.mask_decoders = []
        for i in range(4):
            tin_channels = start_filters * 2 ** (depth - 4 + i)
            decoder = UnetDecoderNotShared(in_channels=tin_channels,
                                         out_channels=out_channels_mask, depth=i + 2,      # 因为上采样层有 i + 2 - 1个
                                         blocks=blocks[i + 8], residual=residual, batch_norm=batch_norm,
                                         transpose=transpose, concat=concat, norm=nn.InstanceNorm2d)
            self.mask_decoders.append(decoder)

        self.wm_decoders = []
        for i in range(4):
            tin_channels = start_filters * 2 ** (depth - 4 + i)
            decoder = UnetDecoderNotShared(in_channels=tin_channels,
                                         out_channels=out_channels_image, depth=i + 2,      # 因为上采样层有 i + 2 - 1个
                                         blocks=blocks[i + 12], residual=residual, batch_norm=batch_norm,
                                         transpose=transpose, concat=concat, norm=nn.InstanceNorm2d)
            self.wm_decoders.append(decoder)

        self.use_coarser = use_coarser
        self.long_skip = long_skip
        self.no_stage2 = no_stage2
        self._forward = self.unshared_forward
        if self.shared != 0:
            self._forward = self.shared_forward
            self.shared_decoder = UnetDecoderShared(in_channels=start_filters * 2 ** (depth - 1),
                                                  out_channels=out_channels_image,
                                                  depth=shared_depth, blocks=blocks[1], residual=residual,
                                                  batch_norm=batch_norm, transpose=transpose, concat=concat,
                                                  is_final=True, norm=nn.InstanceNorm2d)

        if s2am == 'vms2am':
            self.s2am = VMSingleS2AM(4, down=ResDownNew, up=ResUpNew)

    def set_optimizers(self):
        self.optimizer_encoder0 =  torch.optim.Adam(self.encoders[0].parameters(), lr=0.001)
        self.optimizer_encoder1 = torch.optim.Adam(self.encoders[1].parameters(), lr=0.001)
        self.optimizer_encoder2 = torch.optim.Adam(self.encoders[2].parameters(), lr=0.001)
        self.optimizer_encoder3 = torch.optim.Adam(self.encoders[3].parameters(), lr=0.001)
        self.optimizer_image0 =  torch.optim.Adam(self.image_decoders[0].parameters(), lr=0.001)
        self.optimizer_image1 = torch.optim.Adam(self.image_decoders[1].parameters(), lr=0.001)
        self.optimizer_image2 = torch.optim.Adam(self.image_decoders[2].parameters(), lr=0.001)
        self.optimizer_image3 = torch.optim.Adam(self.image_decoders[3].parameters(), lr=0.001)
        self.optimizer_mask0 = torch.optim.Adam(self.mask_decoders[0].parameters(), lr=0.001)
        self.optimizer_mask1 = torch.optim.Adam(self.mask_decoders[1].parameters(), lr=0.001)
        self.optimizer_mask2 = torch.optim.Adam(self.mask_decoders[2].parameters(), lr=0.001)
        self.optimizer_mask3 = torch.optim.Adam(self.mask_decoders[3].parameters(), lr=0.001)
        self.optimizer_wm0 = torch.optim.Adam(self.wm_decoders[0].parameters(), lr=0.001)
        self.optimizer_wm1 = torch.optim.Adam(self.wm_decoders[1].parameters(), lr=0.001)
        self.optimizer_wm2 = torch.optim.Adam(self.wm_decoders[2].parameters(), lr=0.001)
        self.optimizer_wm3 = torch.optim.Adam(self.wm_decoders[3].parameters(), lr=0.001)
        self.optimizer_s2am = torch.optim.Adam(self.s2am.parameters(), lr=0.001)
        if self.shared != 0:
            self.optimizer_shared = torch.optim.Adam(self.shared_decoder.parameters(), lr=0.001)   # 共享参数

    def zero_grad_all(self):
        self.optimizer_encoder0.zero_grad()
        self.optimizer_encoder1.zero_grad()
        self.optimizer_encoder2.zero_grad()
        self.optimizer_encoder3.zero_grad()
        self.optimizer_image0.zero_grad()
        self.optimizer_image1.zero_grad()
        self.optimizer_image2.zero_grad()
        self.optimizer_image3.zero_grad()
        self.optimizer_mask0.zero_grad()
        self.optimizer_mask1.zero_grad()
        self.optimizer_mask2.zero_grad()
        self.optimizer_mask3.zero_grad()
        self.optimizer_wm0.zero_grad()
        self.optimizer_wm1.zero_grad()
        self.optimizer_wm2.zero_grad()
        self.optimizer_wm3.zero_grad()
        self.optimizer_s2am.zero_grad()
        if self.shared != 0:
            self.optimizer_shared.zero_grad()

    def step_all(self):
        self.optimizer_encoder0.step()
        self.optimizer_encoder1.step()
        self.optimizer_encoder2.step()
        self.optimizer_encoder3.step()
        self.optimizer_image0.step()
        self.optimizer_image1.step()
        self.optimizer_image2.step()
        self.optimizer_image3.step()
        self.optimizer_mask0.step()
        self.optimizer_mask1.step()
        self.optimizer_mask2.step()
        self.optimizer_mask3.step()
        self.optimizer_wm0.step()
        self.optimizer_wm1.step()
        self.optimizer_wm2.step()
        self.optimizer_wm3.step()
        self.optimizer_s2am.step()
        if self.shared != 0:
            self.optimizer_shared.step()

    def __call__(self, synthesized):
        return self._forward(synthesized)

    def forward(self, synthesized):
        return self._forward(synthesized)

    def unshared_forward(self, synthesized):  # 当shared_depth = 0
        index = 0
        refines = []
        reconstructed_images = [] # 得到每一个编码器与解码器的图像
        reconstructed_masks = []
        reconstructed_wms = []
        for encoder in self.encoders:
            encoder = encoder.to(synthesized.device)
        for decoder in self.image_decoders:
            decoder = decoder.to(synthesized.device)
        for decoder in self.mask_decoders:
            decoder = decoder.to(synthesized.device)
        for decoder in self.wm_decoders:
            decoder = decoder.to(synthesized.device)

        for encoder in self.encoders:
            image_code, before_pool = encoder(synthesized)
            reconstructed_image = torch.tanh(self.image_decoders[index](image_code, before_pool))
            reconstructed_mask = torch.tanh(self.mask_decoders[index](image_code, before_pool))
            reconstructed_wm = torch.tanh(self.wm_decoders[index](image_code, before_pool))
            if self.long_skip:
                reconstructed_image = reconstructed_image + synthesized
                reconstructed_wm = reconstructed_wm + synthesized
            coarser = reconstructed_image * reconstructed_mask + (1 - reconstructed_mask) * synthesized
            if self.use_coarser:
                refine = torch.tanh(self.s2am(torch.cat([coarser, reconstructed_mask], dim=1))) + coarser
            elif self.no_stage2:
                refine = torch.tanh(self.s2am(torch.cat([coarser, reconstructed_mask], dim=1)))
            else:
                refine = torch.tanh(self.s2am(torch.cat([coarser, reconstructed_mask], dim=1))) + synthesized
            refines.append(refine)
            reconstructed_images.append(reconstructed_image)
            reconstructed_masks.append(reconstructed_mask)
            reconstructed_wms.append(reconstructed_wm)
            index += 1
        # 以下代码按照平均值进行求四个编码-解码分支进行融合
        stacked_refines = torch.stack(refines)
        res = torch.mean(stacked_refines, dim=0)
        stacked_reconstructed_images = torch.stack(reconstructed_images)
        res_image = torch.mean(stacked_reconstructed_images, dim=0)
        stacked_reconstructed_masks = torch.stack(reconstructed_masks)
        res_mask = torch.mean(stacked_reconstructed_masks, dim=0)
        # mask要求[0,1]，因此对其进行缩放
        min_value = torch.min(res_mask)
        max_value = torch.max(res_mask)
        res_mask = (res_mask - min_value) / (max_value - min_value)
        stacked_reconstructed_wms = torch.stack(reconstructed_wms)
        res_wm = torch.mean(stacked_reconstructed_wms, dim=0)
        # return [res, res_image], res_mask, res_wm
        i = 3
        min_value = torch.min(reconstructed_masks[i])
        max_value = torch.max(reconstructed_masks[i])
        reconstructed_masks[i] = (reconstructed_masks[i] - min_value) / (max_value - min_value)

        return [refines[i], reconstructed_images[i]], reconstructed_masks[i], reconstructed_wms[i]

    def shared_forward(self, synthesized):
        # 这里是shared_depth不为0时的代码
        index = 0
        refines = []
        reconstructed_images = []
        reconstructed_masks = []
        reconstructed_vms = []
        for encoder in self.encoders:
            encoder = encoder.to(synthesized.device)
        for decoder in self.image_decoders:
            decoder = decoder.to(synthesized.device)
        for decoder in self.image_decoders:
            decoder = decoder.to(synthesized.device)

        for encoder in self.encoders:
            encoder = encoder.to(synthesized.device)
            print(synthesized.device)
            print(encoder.device)
            image_code, before_pool = encoder(synthesized)
            reconstructed_image = torch.tanh(self.image_decoders[index](image_code, before_pool))
            reconstructed_mask = torch.tanh(self.mask_decoders[index](image_code, before_pool))
            reconstructed_wm = torch.tanh(self.wm_decoders[index](image_code, before_pool))
            if self.long_skip:
                reconstructed_image = reconstructed_image + synthesized
                reconstructed_vm = reconstructed_vm + synthesized
            coarser = reconstructed_image * reconstructed_mask + (1 - reconstructed_mask) * synthesized
            if self.use_coarser:
                refine = torch.tanh(self.s2am(torch.cat([coarser, reconstructed_mask], dim=1))) + coarser
            elif self.no_stage2:
                refine = torch.tanh(self.s2am(torch.cat([coarser, reconstructed_mask], dim=1)))
            else:
                refine = torch.tanh(self.s2am(torch.cat([coarser, reconstructed_mask], dim=1))) + synthesized
            refines.append(refine)
            reconstructed_images.append(reconstructed_image)
            reconstructed_masks.append(reconstructed_mask)
            reconstructed_vms.append(reconstructed_wm)
        res = np.zeros((4, 3, 256, 256))
        res = torch.from_numpy(res)
        res_image = np.zeros((4, 3, 256, 256))
        res_image = torch.from_numpy(res_image)
        res_mask = np.zeros((4, 3, 256, 256))
        res_mask = torch.from_numpy(res_mask)
        res_wm = np.zeros((4, 3, 256, 256))
        res_wm = torch.from_numpy(res_wm)
        for i in range(4):
            res += refines[i]
            res_image += reconstructed_images[i]
            res_mask += reconstructed_masks[i]
            res_wm += reconstructed_vms[i]

        return [res, res_image], res_mask, res_wm