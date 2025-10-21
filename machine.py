import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from torch.autograd import Variable

from math import log10
import time, os

from utils import VGGLoss,is_dic
from torchvision.models import vgg19
from imutils import im_to_numpy
import wm_removers as archs
import RIE_module.common as common
from RIE_module.model import *
from RIE_module.utils import *
from RIE_module.vgg_loss_adv import *

from sklearn.metrics import f1_score

from logger import get_logger

logger = get_logger(name="MyLogger")  # Creating a Logger

class VGGLoss(nn.Module):
    def __init__(self, device):
        super(VGGLoss, self).__init__()
        self.vgg = vgg19(pretrained=True).features[:36].to(device).eval()
        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, input, target):
        vgg_input_features = self.vgg(input)
        vgg_target_features = self.vgg(target)
        loss = F.mse_loss(vgg_input_features, vgg_target_features)
        return loss

'''
    The class for training adversarial image
'''
class Wv(object):
    def __init__(self, datasets =(None,None), models = None, args = None, **kwargs):
        super(Wv, self).__init__()
        self.args = args
        self.device = torch.device(
            self.args.device if self.args and 'device' in self.args else 'cuda' if torch.cuda.is_available() else 'cpu')
        
        mean = (0.0, 0.0, 0.0)
        std = (1.0, 1.0, 1.0)
        mu = torch.tensor(mean).view(3, 1, 1).cuda()
        std = torch.tensor(std).view(3, 1, 1).cuda()
        self.upper_limit = ((1 - mu) / std)
        self.lower_limit = ((0 - mu) / std)

        if self.args.arch is not None:
            # =======================================================================================
            #   Initialize the watermark-removal model
            # =======================================================================================

            # Load the watermark removal model
            if self.args.arch == 'vvv4n':
                # SplitNet
                self.model = archs.__dict__[self.args.arch]()
            else:
                # SLBR, DENet, MNet, etc.
                self.model = archs.__dict__[self.args.arch](args=args)

            self.resume(self.args.resume)
            self.model.to(self.device) 
            self.model.eval() # set model to evaluate

            logger.info(f">>> Initialize the watermark-removal model, {self.args.arch}")
            self.train_loader, self.val_loader = datasets

    def train(self):
        logger.info('>>> Start Attacking')

        configs = ( f'\n'
                    f'\t Attack method: {self.args.attack_method} \n'
                    f'\t Attack the wm network: {self.args.arch} \n'
                    f'\t Total images to generate: {self.args.stopnum} \n'
                    f'\t Dataset: {self.args.data} \n'
                    f'\t Stage 1 configs: \n'
                    f'\t \t epsilon: {self.args.epsilon} / 255 \n'
                    f'\t \t step_alpha: {self.args.step_alpha} / 255 \n'
                    f'\t \t iters: {self.args.iters} \n'
                    f'**if attack method does not contain the RIE module, please ignore the following** \n'
                    f'\t Stage 2 configs: \n'
                    f'\t \t lr: {self.args.lr} \n'
                    f'\t \t Pertubation lr: {self.args.lr2/255} \n'
                    f'\t \t iters: {self.args.rie_iters} \n'
        )                                   

        logger.info(configs)

        dir = self.args.checkpoint          # the directory to save
        os.makedirs(dir, exist_ok=True)
        config_path = os.path.join(dir, 'config.txt')
        with open(config_path, 'a', encoding='utf-8') as f:
            f.write(configs)
        
        attack = Attack(model=self.model, upper_limit=self.upper_limit, lower_limit=self.lower_limit,
                            args=self.args)
        attack.attack(self.val_loader, self.args.data, 'testset')

    def resume(self, resume_path):
        # if isfile(resume_path):
        if not os.path.exists(resume_path):
            resume_path = os.path.join(self.args.checkpoint, 'checkpoint.pth.tar')
        if not os.path.exists(resume_path):
            raise Exception("=> no checkpoint found at '{}'".format(resume_path))

        logger.info(">>> loading checkpoint '{}'".format(resume_path))
        current_checkpoint = torch.load(resume_path)
        if isinstance(current_checkpoint['state_dict'], torch.nn.DataParallel):
            current_checkpoint['state_dict'] = current_checkpoint['state_dict'].module

        if isinstance(current_checkpoint['optimizer'], torch.nn.DataParallel):
            current_checkpoint['optimizer'] = current_checkpoint['optimizer'].module

        if self.args.start_epoch == 0:
            self.args.start_epoch = current_checkpoint['epoch']
        self.metric = current_checkpoint['best_acc']
        items = list(current_checkpoint['state_dict'].keys())

        ## restore the learning rate
        lr = self.args.lr
        for epoch in self.args.schedule:
            if epoch <= self.args.start_epoch:
                lr *= self.args.gamma
        optimizers = [getattr(self.model, attr) for attr in dir(self.model) if
                      attr.startswith("optimizer") and getattr(self.model, attr) is not None]
        for optimizer in optimizers:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        # ---------------- Load Model Weights --------------------------------------
        self.model.load_state_dict(current_checkpoint['state_dict'], strict=False)          

        logger.info(">>> loaded checkpoint '{}' (epoch {})"
              .format(resume_path, current_checkpoint['epoch']))


'''
    The class for attacking the watermark-removal model and generating adversarial images
'''
class Attack:
    def __init__(self, model, upper_limit=None, lower_limit=None, args = None):
        # Initialize the attack parameters
        self.model = model
        # --------------------------------------------
        #     First stage: train the perturbation
        #     Second stage: train the adversarial image using the RIE module
        # --------------------------------------------
        self.epsilon = args.epsilon / 255.0                         # Epsilon for the first stage
        self.step_alpha = args.step_alpha / 255.0                   # Step alpha for the first stage

        self.iters = args.iters                                     # Iterations for the first stage    
        self.rie_iters = args.rie_iters                             # Iterations for the second stage

        self.lambda_p = args.lambda_p                               # Perception loss weight, default 200

        self.upper_limit = upper_limit
        self.lower_limit = lower_limit
        self.early_stop_tolerance = 10  
        self.early_stop_threshold = 0.01  

        self.args = args
        self.now = None
        self.device = torch.device(
            self.args.device if self.args and 'device' in self.args else 'cuda' if torch.cuda.is_available() else 'cpu')
        self.vgg_loss = VGGLoss(self.device).to(self.device)

        if self.args.use_rie:
            print(self.args.use_rie)
            print(type(self.args.use_rie))

            logger.info('>>> THE ATTACKING METHOD CONTAINS THE RIE MODULE')
            # ======================================================
            #    Network initialization
            # ======================================================
            self.rie_module = Model().to(self.device)
            init_model(self.rie_module)
            self.rie_module = torch.nn.DataParallel(self.rie_module, device_ids=[0])

            para = get_parameter_number(self.rie_module)
            logger.info(f'\tThe RIE Module parameters: {para}')

            params_trainable = (list(filter(lambda p: p.requires_grad, self.rie_module.parameters())))

            self.optimizer_rie = torch.optim.Adam(params_trainable, lr=self.args.lr, betas=(0.5, 0.999), eps=1e-6, weight_decay=self.args.weight_decay)
            self.weight_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer_rie, self.args.weight_step, gamma=self.args.gamma)
            self.optim_init = self.optimizer_rie.state_dict()
            # ======================================================
            #   DWT and IWT initialization
            # ======================================================
            self.dwt = common.DWT()
            self.iwt = common.IWT()

            self.vgg_loss_adv = VGGLossadv(3, 1, False).to(self.device)

    def pgd_attack(self, inputs, target):
        # ======================================================================
        #   PGD attack
        #   Can be used in generating adversarial perturbation directly
        # ======================================================================
        r = torch.zeros_like(inputs).cuda()
        r.requires_grad = True
        for i in range(self.iters):
            input_new = inputs + r
            outputs = self.model(self.norm(input_new)) 
            if self.args.arch == 'vvv4n':
                imoutput, immask, imwatermark = outputs
                imoutput = imoutput[0] if is_dic(imoutput) else imoutput
            elif self.args.arch == 'slbr':
                imoutput, immask, imwatermark = outputs
                immask = immask[0]
                imoutput = imoutput[0] if is_dic(imoutput) else imoutput
            elif self.args.arch == 'denet':
                imoutput, immask_all, _, _ = outputs
                imoutput = imoutput[0] if is_dic(imoutput) else imoutput
                immask = immask_all[0]
            else:
                imoutput, immask = outputs
            imfinal2 = imoutput * immask + input_new * (1 - immask)

            loss = F.mse_loss(imfinal2, target, reduction='sum')

            loss.backward()

            with torch.no_grad():
                grad = r.grad.detach()
                d = r
                d = torch.clamp(d + self.step_alpha * torch.sign(grad), -self.epsilon, self.epsilon)
                r.data = d
                r.grad.zero_()
        pertubation = r.detach()
        r = r.detach() + inputs
        return r, pertubation

    def rie_module_attack(self, wm_i, ori_i, x=None):
        # ======================================================
        #       wm_i:   the watermarked image
        #       ori_i:   the original image
        #       x:        the pertubation image
        # ======================================================
        # Set up the initial perturbation image
        #   If no perturbation image is specified, 
        #       a random noise is generated directly according to the epsilon size
        if x is None:
            delta = torch.rand(wm_i.size()).to(self.device)
            delta = torch.clamp(delta, -self.epsilon, self.epsilon)
        else:
            delta = x.to(self.device)
        delta_ori = delta.to(self.device)
        delta_ori = Variable(delta_ori, requires_grad=True)
        # ======================================================
        # Sets the optimizer used to further optimize the perturbation.
        optimizer_delta = torch.optim.Adam([delta_ori], lr=self.args.lr2/255)
        # ======================================================
        # Convert watermarked image to frequency domain
        wm_i_dwt =  self.dwt(wm_i).to(self.device)
        # ======================================================
        # Start for iterating
        for i in range(self.rie_iters):
            # ============================================================================================================
            # Forward process
            delta_varia = delta_ori.to(self.device)                                 # (1, 3, 256, 256)
            delta_varia_dwt = self.dwt(delta_varia).to(self.device)                 # (1, 12, 128, 128)

            input_dwt = torch.cat((wm_i_dwt, delta_varia_dwt), 1).to(self.device)   # (1, 24, 128, 128)
            output_dwt = self.rie_module(input_dwt).to(self.device)                 # (1, 24, 128, 128)
            adv_i_dwt = output_dwt.narrow(1, 0, 4 * self.args.channels_in).to(self.device)                          # (1, 12, 128, 128)
            r_i_dwt = output_dwt.narrow(1, 4 * self.args.channels_in, 4 * self.args.channels_in).to(self.device)    # (1, 12, 128, 128)

            adv_i = self.iwt(adv_i_dwt).to(self.device)                            # (1, 3, 256, 256)
            r_i = self.iwt(r_i_dwt).to(self.device)                                # (1, 3, 256, 256)
            # ============================================================================================================
            # Backward process
            back_input_dwt = torch.cat((adv_i_dwt, r_i_dwt), 1).to(self.device)    # the same as 'output_dwt'
            back_output_dwt = self.rie_module(back_input_dwt, rev=True)

            back_wm_i_dwt = back_output_dwt.narrow(1, 0, 4 * self.args.channels_in).to(self.device)                             # watermarked image in backward process
            back_delta_dwt = back_output_dwt.narrow(1, 4 * self.args.channels_in, 4 * self.args.channels_in).to(self.device)    # pertubation image in backward process

            back_delta = self.iwt(back_delta_dwt).to(self.device)
            # ============================================================================================================
            # Loss for optimize the perceptual quality of the adversarial image
            loss_perc = guide_loss(adv_i.cuda(), wm_i.cuda()).to(self.device) 
            # ============================================================================================================
            # Input the adversarial image into the watermark removal network
            outputs = self.model(adv_i)

            if self.args.arch == 'splitnet':       
                # SplitNet
                imoutput, immask, _ = outputs
                imoutput = imoutput[0] if is_dic(imoutput) else imoutput
            elif self.args.arch == 'denet':
                # DENet
                imoutput, immask_all, _, _ = outputs
                imoutput = imoutput[0] if is_dic(imoutput) else imoutput
                immask = immask_all[0]
            elif self.args.arch == 'slbr':
                # SLBR
                imoutput, immask, _ = outputs
                immask = immask[0]
                imoutput = imoutput[0] if is_dic(imoutput) else imoutput
            else:
                # Other watermark removal models, e.g. MNet
                imoutput, immask = outputs              # You can modify this part to suit your specific model
            wm_removed_i = imoutput * immask + adv_i * (1 - immask)  # watermark removed image
            # ============================================================================================================
            # Loss for optimize the RIE-module
             
            loss_adv = F.mse_loss(wm_removed_i, ori_i, reduction='sum')

            total_loss = (-1) * loss_adv + self.lambda_p * loss_perc

            self.optimizer_rie.zero_grad()
            optimizer_delta.zero_grad()

            total_loss.backward()
            self.optimizer_rie.step()
            # ============================================================================================================
            # Further optimization of the perturbation, in the backward process
            back_delta = Variable(back_delta, requires_grad=True)
            adv_i_2 = back_delta + wm_i   # Add to watermarked image directly

            outputs = self.model(self.norm(adv_i_2))
            if self.args.arch == 'vvv4n':
                imoutput, immask, _ = outputs
                imoutput = imoutput[0] if is_dic(imoutput) else imoutput
            elif self.args.arch == 'denet':
                imoutput, immask_all, _, _ = outputs
                imoutput = imoutput[0] if is_dic(imoutput) else imoutput
                immask = immask_all[0]
            elif self.args.arch == 'slbr':
                imoutput, immask, _ = outputs
                immask = immask[0]
                imoutput = imoutput[0] if is_dic(imoutput) else imoutput
            else:
                imoutput, immask = outputs
            wm_removed_i_2 = imoutput * immask + adv_i_2 * (1 - immask)
            loss_back = (-1) * F.mse_loss(wm_removed_i_2, ori_i, reduction='sum')

            loss_back.backward()
            optimizer_delta.step()
            # ============================================================================================================
            self.weight_scheduler.step()
            # ============================================================================================================
            # logging
            if i % 50 == 0:
                logger.loss_print(f'\tRIE Module Attack Iteration {i}/{self.rie_iters}, Total Loss: {total_loss.item()},'
                            f' loss_adv: {loss_adv.item()}, loss_perc: {loss_perc.item()}')
            # ============================================================================================================
        return adv_i, r_i  # adv_i: the adversarial image, r_i: the redundant image
    
    def train_perturbation(self, inputs=None, target=None, mask=None, wm=None):
        if self.args.attack_method == 'pgd':
            r, _ = self.pgd_attack(self.norm(inputs), self.norm(target))
        elif self.args.attack_method == 'pgd_inn':
            # The followings are using watermarked images for attacking
            _, x1  = self.pgd_attack(self.norm(inputs), self.norm(target))              # First stage: PGD attack
            r, xr = self.rie_module_attack(self.norm(inputs), self.norm(target), x1)    # Second stage: RIE module attack
        # You can add other attack methods here

        adv_image = self.norm(r)  # is adversarial image
    
        return adv_image

    def get_output(self, clean_image=None, adv_image=None):
        # ===========================================================================================
        #   Get the watermark-removed output of the clean and adversarial images
        # ===========================================================================================
        if self.args.arch == 'vvv4n':
            # SplitNet
            # =======================================================================================
            clean_outputs = self.model(clean_image)
            clean_imoutput, clean_immask, _ = clean_outputs
            clean_imoutput = clean_imoutput[0] if is_dic(clean_imoutput) else clean_imoutput
            # =======================================================================================
            adv_outputs = self.model(adv_image)
            adv_imoutput, adv_immask, _ = adv_outputs
            adv_imoutput = adv_imoutput[0] if is_dic(adv_imoutput) else adv_imoutput
            # =======================================================================================
        elif self.args.arch == 'slbr':
            # SLBR
            # =======================================================================================
            clean_outputs = self.model(clean_image)
            clean_imoutput, clean_immask, _ = clean_outputs
            clean_immask = clean_immask[0]
            clean_imoutput = clean_imoutput[0] if is_dic(clean_imoutput) else clean_imoutput
            # =======================================================================================
            adv_outputs = self.model(adv_image)
            adv_imoutput, adv_immask, _ = adv_outputs
            adv_immask = adv_immask[0]
            adv_imoutput = adv_imoutput[0] if is_dic(adv_imoutput) else adv_imoutput
            # =======================================================================================
        elif self.args.arch == 'denet':
            # DENet
            # =======================================================================================
            clean_outputs = self.model(clean_image)
            clean_imoutput, clean_immask_all, _, _ = clean_outputs
            clean_imoutput = clean_imoutput[0] if is_dic(clean_imoutput) else clean_imoutput
            clean_immask = clean_immask_all[0]
            # =======================================================================================
            adv_outputs = self.model(adv_image)
            adv_imoutput, adv_immask_all, _, _ = adv_outputs
            adv_imoutput = adv_imoutput[0] if is_dic(adv_imoutput) else adv_imoutput
            adv_immask = adv_immask_all[0]
            # =======================================================================================
        else:
            # Other watermark removal models, not specified
            # =======================================================================================
            clean_outputs = self.model(clean_image)
            clean_imoutput, clean_immask = clean_outputs
            # =======================================================================================
            adv_outputs = self.model(adv_image)
            adv_imoutput, adv_immask = adv_outputs
        # ===========================================================================================
        clean_output = clean_imoutput * clean_immask + clean_image * (1 - clean_immask)
        adv_output = adv_imoutput * adv_immask + adv_image * (1 - adv_immask)
        return clean_output, adv_output

    def save_imgs(self, clean_image=None, adv_image=None, clean_out=None, adv_out=None, i=None, ori_image=None):
        # =================================================================================================
        #   Save the images for visualization
        # =================================================================================================
        adv_image = im_to_numpy(torch.clamp(adv_image[0].detach() * 255, min=0.0, max=255.0)).astype(np.uint8)
        clean_image = im_to_numpy(torch.clamp(clean_image[0].detach() * 255, min=0.0, max=255.0)).astype(np.uint8)
        clean_out = im_to_numpy(torch.clamp(clean_out[0].detach() * 255, min=0.0, max=255.0)).astype(np.uint8)
        adv_out = im_to_numpy(torch.clamp(adv_out[0].detach() * 255, min=0.0, max=255.0)).astype(np.uint8)
        ori_image = im_to_numpy(torch.clamp(ori_image[0].detach() * 255, min=0.0, max=255.0)).astype(np.uint8)
        dir = self.args.checkpoint
        if not os.path.exists(dir):
            os.makedirs(dir)
        from PIL import Image
        adv_saving_path = os.path.join(dir, 'adv_image')
        clean_saving_path = os.path.join(dir, 'clean_image')
        adv_out_saving_path = os.path.join(dir, 'adv_out')
        clean_out_saving_path = os.path.join(dir, 'clean_out')
        ori_image_saving_path = os.path.join(dir, 'ori_image')
        if not os.path.exists(adv_saving_path):
            os.makedirs(adv_saving_path)
        if not os.path.exists(clean_saving_path):
            os.makedirs(clean_saving_path)
        if not os.path.exists(adv_out_saving_path):
            os.makedirs(adv_out_saving_path)
        if not os.path.exists(clean_out_saving_path):
            os.makedirs(clean_out_saving_path)
        if not os.path.exists(ori_image_saving_path):
            os.makedirs(ori_image_saving_path)
        name = str(i) + '.png'
        adv_image_s = Image.new('RGB', (256, 256))
        adv_image_s.paste(Image.fromarray(adv_image), (0, 0))
        clean_image_s = Image.new('RGB', (256, 256))
        clean_image_s.paste(Image.fromarray(clean_image), (0, 0))
        clean_out_s = Image.new('RGB', (256, 256))
        clean_out_s.paste(Image.fromarray(clean_out), (0, 0))
        adv_out_s = Image.new('RGB', (256, 256))
        adv_out_s.paste(Image.fromarray(adv_out), (0, 0))
        ori_image_s = Image.new('RGB', (256, 256))
        ori_image_s.paste(Image.fromarray(ori_image), (0, 0))

        adv_image_s.save('%s/%s' % (adv_saving_path, name))
        logger.info('\tsaved adv image: ' + adv_saving_path + ' | ' + name)
        clean_image_s.save('%s/%s' % (clean_saving_path, name))
        logger.info('\tsaved clean image: ' + clean_saving_path + ' | ' + name)
        adv_out_s.save('%s/%s' % (adv_out_saving_path, name))
        logger.info('\tsaved adv out: ' + adv_out_saving_path + ' | ' + name)
        clean_out_s.save('%s/%s' % (clean_out_saving_path, name))
        logger.info('\tsaved clean out: ' + clean_out_saving_path + ' | ' + name)
        ori_image_s.save('%s/%s' % (ori_image_saving_path, name))
        logger.info('\tsaved original image: ' + ori_image_saving_path + ' | ' + name)
        
        return True

    def get_psnr(self, image1=None, image2=None):
        mse_loss = F.mse_loss(image1, image2).item()
        if mse_loss == 0:
            psnr = 100
        else:
            psnr = 10 * log10(1 / (mse_loss))
        return psnr

    def attack_success_check(self, psnr1=None, psnr2=None):
        # ======================================================================================================================
        # To check whether the attack is successful
        #   psnr1: PSNR between clean output and original image
        #   psnr2: PSNR between adv output and original image
        #   You can modify the attack success condition here
        # ======================================================================================================================
        a_s = 1.0
        a_f = 0.0
        if psnr1 - psnr2 >= 15 or psnr2 <= 20:
            return a_s
        else:
            return a_f

    def attack(self, val_loader, tdataset, tdata):
        # ===================================================================================================================
        # To generate the adversarial examples
        # ===================================================================================================================
        psnres_1 = []
        psnres_2 = []
        psnres_3 = []
        cheak_attacks = []

        tdataset = tdataset
        tdata = tdata
        tdatanum = 0
        attack_sucess = 0
        start_time = time.time()
        clean_samples = []
        attack_samples = []
        targets = []
        masks = []
        for i, batches in enumerate(val_loader):
            # Stop Num
            if tdatanum >= self.args.stopnum:
                break
            tdatanum = tdatanum + 1
            # Read Data
            inputs = batches['image'].float().cuda()
            target = batches['target'].float().cuda()
            mask = batches['mask'].float().cuda()
            wm = batches['wm'].float().cuda()
            # get adv image
            adv_image = self.train_perturbation(inputs, target, mask, wm) # adv image
            # get clean image
            clean_image = self.norm(inputs)  # not adv

            clean_samples.append(clean_image)
            attack_samples.append(adv_image)
            targets.append(target)
            masks.append(mask)

            # Get output from watermark removal model when input clean and adv image
            clean_out, adv_out = self.get_output(clean_image, adv_image)

            # Saving images
            if self.args.simage:
                simage = self.save_imgs(clean_image, adv_image, clean_out, adv_out, i, target)

            # Evaluation
            logger.info('>>> NOW EVALUATING THE ATTACKING METHOD')

            psnr_impec = self.get_psnr(clean_image, adv_image)          # PSNR between adv image and watermarked image
            psnr_ori_out = self.get_psnr(clean_out, target)             # PSNR between clean output and original image
            psnr_adv_out = self.get_psnr(adv_out, target)               # PSNR between adv output and original image

            psnres_1.append(psnr_impec) 
            psnres_2.append(psnr_ori_out)
            psnres_3.append(psnr_adv_out)

            if psnr_impec == 100:
                # Avoid outliers
                continue

            cheak_attack_success = self.attack_success_check(psnr_ori_out, psnr_adv_out)
            cheak_attacks.append(cheak_attack_success)
            attack_sucess = attack_sucess + cheak_attack_success

            logger.metric_print(
                f"\tSingle Value (PSNR, for this sample): adv image and original image - {psnr_impec:.5f} dB, original output: {psnr_ori_out:.5f} dB, adv output: {psnr_adv_out:.5f} dB")
            logger.metric_print(f"\tHave processed {tdatanum} images")
            logger.metric_print(f"\tMean Value (PSNR, for all processed samples): adv image and original image - {np.mean(psnres_1):.5f} dB, original output: {np.mean(psnres_2):.5f} dB, adv output: {np.mean(psnres_3):.5f} dB")
            logger.metric_print(
                f"\tStd Value (PSNR, for all processed samples): adv image and original image - {np.std(psnres_1):.5f} dB, original output: {np.std(psnres_2):.5f} dB, adv output: {np.std(psnres_3):.5f} dB")
            logger.metric_print(
                f"\tMean Value (ASR, for all processed samples): mean asr - {np.mean(cheak_attacks):.5f}")
            logger.metric_print(
                f"\tStd Value (ASR, for all processed samples): std asr - {np.std(cheak_attacks):.5f}")
            y_true = [1] * len(cheak_attacks) # caculate F1
            asr_f1 = f1_score(y_true, cheak_attacks)
            logger.metric_print(
                f"\tF1 (ASR, for all processed samples): F1 asr - {asr_f1:.5f}")

            del inputs, target, mask, wm, clean_image, adv_image, clean_out, adv_out

        datas = {
            'attack_method': self.args.attack_method,
            'dataset': self.args.data,
            'model1': self.args.arch,
            'PSNR_impec_mean':round(np.mean(psnres_1),5),
            'PSNR_impec_std': round(np.std(psnres_1), 5),
            'PSNR_ori_mean': round(np.mean(psnres_2), 5),
            'PSNR_ori_std': round(np.std(psnres_2), 5),
            'PSNR_adv_mean': round(np.mean(psnres_3), 5),
            'PSNR_adv_std': round(np.std(psnres_3), 5),
            'asr_mean': round(np.mean(cheak_attacks), 5),
            'asr_std': round(np.std(cheak_attacks), 5),
            'f1': round(asr_f1, 5)
        }
        print('=================FINAL RESULTS=================')
        print(datas)

    def norm(self,x):
        if self.args.gan_norm:
            return x*2.0 - 1.0
        else:
            return x

    def denorm(self,x):
        if self.args.gan_norm:
            return (x+1.0)/2.0
        else:
            return x