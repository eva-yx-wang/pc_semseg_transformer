import __init__
import os, argparse, yaml, numpy as np
from torch import multiprocessing as mp
from examples.classification.train import main as train
from examples.classification.pretrain import main as pretrain
from openpoints.utils import EasyConfig, dist_utils, find_free_port, generate_exp_directory, resume_exp_directory, Wandb
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(ROOT_DIR, "log")

if __name__ == "__main__":
    parser = argparse.ArgumentParser('S3DIS scene segmentation training')
    parser.add_argument('--cfg', type=str, required=True, help='config file')
    #parser.add_argument('--gpu', default="0", type=str, required=True, help='')
    parser.add_argument('--profile', action='store_true', default=False, help='set to True to profile speed')
    args, opts = parser.parse_known_args()


    '''DEVICES'''
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"#str(args.gpu)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # log level: 2--only show warning and Error; 3--only show error
    cfg = EasyConfig()
    cfg.load(args.cfg, recursive=True)
    cfg.update(opts)  #?
    if cfg.seed is None:
        cfg.seed = np.random.randint(1, 10000)

    # init distributed env first, since logger depends on the dist info.
    cfg.rank, cfg.world_size, cfg.distributed, cfg.mp = dist_utils.get_dist_info(cfg)
    cfg.sync_bn = cfg.world_size > 1

    # init log dir
    cfg.task_name = args.cfg.split('.')[-2].split('/')[-2]  # dataset name
    cfg.exp_name = args.cfg.split('.')[-2].split('/')[-1]  # exp model name
    tags = [
        cfg.task_name,  # task name (the folder of name under ./cfgs
        cfg.mode,
        cfg.exp_name,  # cfg file name
        f'ngpus{cfg.world_size}',
        f'seed{cfg.seed}',
    ]
    for i, opt in enumerate(opts):
        if 'rank' not in opt and 'dir' not in opt and 'root' not in opt and 'path' not in opt and 'wandb' not in opt and '/' not in opt:
            tags.append(opt)
    #cfg.root_dir = os.path.join(cfg.root_dir, cfg.task_name)
    cfg.root_dir = os.path.join(LOG_DIR, cfg.task_name)

    if not cfg.mode == 'resume':
        generate_exp_directory(cfg, tags, additional_id=os.environ.get('MASTER_PORT', None))
        cfg.wandb.tags = tags
    else:  # resume from the existing ckpt and reuse the folder.
        resume_exp_directory(cfg, pretrained_path=cfg.pretrained_path)
        cfg.wandb.tags = ['resume']
    os.environ["JOB_LOG_DIR"] = cfg.log_dir
    cfg_path = os.path.join(cfg.run_dir, "cfg.yaml")
    with open(cfg_path, 'w') as f:
        yaml.dump(cfg, f, indent=2)
        os.system('cp %s %s' % (args.cfg, cfg.run_dir))
    cfg.cfg_path = cfg_path
    cfg.wandb.name = cfg.run_name

    if cfg.mode == 'pretrain':
        main = pretrain
    else:
        main = train

    # multi processing.
    if cfg.mp:
        port = find_free_port()
        cfg.dist_url = f"tcp://localhost:{port}"
        print('using mp spawn for distributed training')
        mp.spawn(main, nprocs=cfg.world_size, args=(cfg, args.profile))
    else:
        main(0, cfg, profile=args.profile)  #3