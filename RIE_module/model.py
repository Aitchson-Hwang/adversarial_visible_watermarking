import torch.optim
import torch.nn as nn
from RIE_module.dense import Dense

# =======================================================================================
# Building of the RIE Module
# =======================================================================================
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.model = affine_blocks()

    def forward(self, x, rev=False):

        if not rev:
            out = self.model(x)

        else:
            out = self.model(x, rev=True)

        return out

def init_model(mod):
    for key, param in mod.named_parameters():
        split = key.split('.')
        if param.requires_grad:
            param.data = 0.01 * torch.randn(param.data.shape).cuda()
            if split[-2] == 'conv5':
                param.data.fill_(0.)

# =======================================================================================
# Architecture of the RIE Module
# =======================================================================================
class affine_blocks(nn.Module):
    '''Residual in Residual Dense Block'''

    def __init__(self):
        super(affine_blocks, self).__init__()

        # With 3 affine blocks
        self.inv1 = affine_block()
        self.inv2 = affine_block()
        self.inv3 = affine_block()

    def forward(self, x, rev=False):
        if rev:     # Backward process
            out = self.inv1(x, back=True)
            out = self.inv2(out, back=True)
            out = self.inv3(out, back=True)
            return out
        else:       # Forward process
            out = self.inv1(x)
            out = self.inv2(out)
            out = self.inv3(out)
        return out


# =======================================================================================
# Invertible Affine Coupling block
# =======================================================================================
class affine_block(nn.Module):
    def __init__(self, subnet_constructor=Dense, clamp=2.0, harr=True, x_1=3, x_2=3):
        super().__init__()
        if harr:
            self.x_len1 = x_1*4
            self.x_len1 = x_2*4
        self.clamp = clamp
        # \rho
        self.r = subnet_constructor(self.x_len1, self.x_len1)
        # \gamma
        self.g = subnet_constructor(self.x_len1, self.x_len1)
        # \mu
        self.m = subnet_constructor(self.x_len1, self.x_len1)
        # \phi 
        self.p = subnet_constructor(self.x_len1, self.x_len1)

    def exp_and_k(self, s):
        return torch.exp(self.clamp * 2 * (torch.sigmoid(s) - 0.5))

    def forward(self, x, back=False):
        if back:    # Backward process
            y1, y2 = (x.narrow(1, 0, self.x_len1),
                      x.narrow(1, self.x_len1, self.x_len1))
            for_dot_1, for_add_1 = self.r(y1), self.g(y1)
            x2 = (y2 - for_add_1) / self.exp_and_k(for_dot_1)
            for_dot_2, for_add_2 = self.p(x2), self.m(x2)
            x1 = (y1 - for_add_2) / self.exp_and_k(for_dot_2)
            return torch.cat((x1, x2), 1)
        else:       # Forward process
            x1, x2 = (x.narrow(1, 0, self.x_len1),
                      x.narrow(1, self.x_len1, self.x_len1))
            for_dot_2, for_add_2 = self.p(x2), self.m(x2)
            y1 = self.exp_and_k(for_dot_2) * x1 + for_add_2
            for_dot_1, for_add_1 = self.r(y1), self.g(y1)
            y2 = self.exp_and_k(for_dot_1) * x2 + for_add_1
        return torch.cat((y1, y2), 1)