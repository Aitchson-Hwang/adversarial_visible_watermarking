
import wm_removers as models

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))
    
class Options():
    """docstring for Options"""
    def __init__(self):
        pass

    def init(self, parser):        
        parser.add_argument('--arch', '-a', metavar='ARCH', default='dhn',
                            choices=model_names,
                            help='model architecture: ' +
                                ' | '.join(model_names) +
                                ' (default: resnet18)')  
        parser.add_argument('--machine', '-m', metavar='NACHINE', default='basic')  
        parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                            help='number of data loading workers (default: 4)')    
        parser.add_argument('--epochs', default=30, type=int, metavar='N',
                            help='number of total epochs to run')                   
        parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                            help='manual epoch number (useful on restarts)')
        parser.add_argument('--train-batch', default=128, type=int, metavar='N',     
                            help='train batchsize')
        parser.add_argument('--test-batch', default=6, type=int, metavar='N',
                            help='test batchsize')   
        parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,metavar='LR', help='initial learning rate') 

        parser.add_argument('--schedule', type=int, nargs='+', default=[5, 10],
                            help='Decrease learning rate at these epochs.')     
        parser.add_argument('--gamma', type=float, default=0.1,
                            help='LR is multiplied by gamma on schedule.')     
        parser.add_argument('--lambda_p', type=float, default=200, help='the weight of perceptual loss.')  
        parser.add_argument('--epsilon', type=float, help='the bound of perturbation', default=8)   
        parser.add_argument('--step_alpha', type=float, help='step size', default=2)    
        parser.add_argument('--iters', type=int, help='attack iters', default=50)   
        parser.add_argument('--use_rie', type=bool, help='use inn to insert perturbation or not', default=0)  
        parser.add_argument('--rie_iters', type=int, help='training iters for the RIE modules', default=50) 
        parser.add_argument('--stopnum', type=int, help='attack iters', default=500000)    
        parser.add_argument('--attack_method', default='pgd', type=str, help='attack method')   
        parser.add_argument('--weight_decay', default=1e-5, type=float, help='weight decay')   
        parser.add_argument('--weight_step', default=200, type=int, help='weight step')
        parser.add_argument('--lamda_low_frequency', default=1, type=float, help='balance losses')    
        parser.add_argument('--lamda_adv', default=3, type=float, help='balance losses')    
        parser.add_argument('--lr2', default=1, type=float, help='initial learning rate for optimizing the perturbation')  
        parser.add_argument('--channels_in', default=3, type=int, help='the input channels') 
        parser.add_argument('--base-dir', default='/PATH_TO_DATA_FOLDER/', type=str, metavar='PATH')   
        parser.add_argument('--data', default='', type=str, metavar='PATH',
                            help='dataset')   
        parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                            help='path to save checkpoint (default: checkpoint)')   
        parser.add_argument('--resume', default='', type=str, metavar='PATH',
                            help='path to latest checkpoint (default: none)')   
        parser.add_argument('--normalized-input', default=False, type=bool,
                            help='whether to normalize the input images')
        parser.add_argument('-da', '--data-augumentation', default=False, type=bool,
                            help='whether to use data augumentation')
        parser.add_argument('--simage', default=False, type=bool, help='save images')   
        parser.add_argument('--input-size', default=256, type=int, metavar='N',
                            help='train batchsize')   
        parser.add_argument('--limited-dataset', default=0, type=int, metavar='N')   
        parser.add_argument('--gan-norm', default=False,type=bool, help='use gan norm or not')
        # parsers for SLBR
        parser.add_argument('--bg_mode', default='res_mask',type=str, help='necessary for SLBR') # vanilla, res_mask, res_feat, proposed
        parser.add_argument('--use_refine', action='store_true', help='necessary for SLBR')
        parser.add_argument('--k_refine', default=3, type=int, help='necessary for SLBR')
        parser.add_argument('--k_skip_stage', default=3, type=int, help='necessary for SLBR')
        return parser