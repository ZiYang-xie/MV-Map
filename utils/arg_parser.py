import argparse
import mmcv

VERSION = 'trainval'
def build_args():
    parser = argparse.ArgumentParser(description='NeuralMap')
    # DDP training
    parser.add_argument('--world-size', default=-1, type=int, 
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int, 
                        help='node rank for distributed training')
    parser.add_argument('--dist_url', default='env://',
                    help="init-method")
    parser.add_argument('--gpu', default=None, type=int)
    parser.add_argument("--multicard", action='store_true')
    parser.add_argument("--local_rank", default=-1)

    parser.add_argument('--config', type=str, help='Exp configs', default='./configs/default.py')
    # nuScenes config
    parser.add_argument('--nusc_dataroot', type=str, default=f'./data/nuScenes/{VERSION}')
    parser.add_argument('--nerf_dataroot', type=str, default=f'./data/nerf_data_{VERSION}/')
    parser.add_argument('--hdmap_dataroot', type=str, default=f'./data/hdmap_{VERSION}')
    parser.add_argument('--nusc_version', type=str, default=f'v1.0-{VERSION}', choices=['v1.0-trainval', 'v1.0-mini'])

    # debug config
    parser.add_argument("--test", action='store_true')
    parser.add_argument("--resume", action='store_true')
    parser.add_argument("--presave", action='store_true')
    parser.add_argument("--debug", action='store_true') # Mini data version
    parser.add_argument("--load_pretrained", action='store_true')

    # training config
    parser.add_argument("--nepochs", type=int, default=16)
    parser.add_argument("--fuser_nepochs", type=int, default=5)
    parser.add_argument("--fusion_accum_steps", type=int, default=5)
    parser.add_argument("--eval_nerf_epoch", type=int, default=30)
    parser.add_argument("--logdir", type=str, default='./logs')
    parser.add_argument("--max_grad_norm", type=float, default=5.0)
    parser.add_argument("--bsz", type=int, default=4)
    parser.add_argument("--nworkers", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-7)
    parser.add_argument("--pretrain_scene", type=str, default='scene-0655')

    # log config
    parser.add_argument("--nerf_sub_iter", type=int, default=500)
    parser.add_argument("--pretrain_iter", type=int, default=30000)
    
    parser.add_argument("--render_interval", type=int, default=9999)
    parser.add_argument("--update_interval", type=int, default=1000)
    parser.add_argument("--eval_interval", type=int, default=5)
    parser.add_argument("--export_interval", type=int, default=9999)
    parser.add_argument("--log_iou", type=int, default=10)
    parser.add_argument("--log_pretrain_interval", type=int, default=500)


    args = parser.parse_args()
    cfg = mmcv.Config.fromfile(args.config)
    return args, cfg