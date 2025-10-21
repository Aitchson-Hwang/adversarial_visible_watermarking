

from wm_removers.slbr import SLBR
from wm_removers.denet import DENet



# MNet
'''
    W. Huang, Y. Dai, J. Fei, and F. Huang, “MNet: A multi-scale network
    for visible watermark removal,” Neural Netw., vol. 183, Mar. 2025, Art.
    no. 106961.
''' 
def mnet(**kwargs):
    return MNet(wf=54, scale=42, vscale=42, args=kwargs['args'])

# SplitNet
'''
    X. Cun and C.-M. Pun, “Split then refine: Stacked attention-guided
    resunets for blind single image visible watermark removal,” in Proc.
    AAAI Conf. Artif. Intell., 2021, pp. 1184-1192.
''' 
def vvv4n(**kwargs):
    return UnetVMS2AMv4(shared_depth=2, blocks=3, long_skip=True, use_vm_decoder=True,s2am='vms2am')

# SLBR
'''
    J. Liang, L. Niu, F. Guo, T. Long, and L. Zhang, “Visible watermark
    removal via self-calibrated localization and background refinement,” in
    Proc. 29th ACM Int. Conf. Multimedia, Oct. 2021, pp. 4426-4434.
''' 
def slbr(**kwargs):
    # print(args = kwargs['args'])
    # return SLBR(shared_depth=1, blocks=3, long_skip=True)
    return SLBR(args=kwargs['args'], shared_depth=1, blocks=3, long_skip=True)

# DENet
'''
    R. Sun, Y. Su, and Q. Wu, “DENet: Disentangled embedding network
    for visible watermark removal,” in Proc. AAAI Conf. Artif. Intell., 2023,
    vol. 37, no. 2, pp. 2411-2419.
''' 
def denet(**kwargs):
    return DENet(args=kwargs['args'], shared_depth=1, blocks=3, long_skip=True)


# BVMR
'''
    A. Hertz, S. Fogel, R. Hanocka, R. Giryes, and D. Cohen-Or, “Blind
    visual motif removal from a single image,” in Proc. IEEE Conf. Comput.
    Vis. Pattern Recognit., Jun. 2019, pp. 6858-6867.
''' 
def vm3(**kwargs):
    return UnetVM(shared_depth=2, blocks=3, use_vm_decoder=True)

