'''
Tools to parse args and merge with yaml config. 

YACS is another choice but is not used due to its complex argument naming in CLI. 

Note:
- To make an argument changable by command line input, claim the arg in `create_parse()` and update the merge rule in `merge_args_to_config()`, update the default value overwrite rule in `_update_parser_default_values_by_config()`. 
'''

import argparse
import yaml


def create_parser():
    # The first arg parser parses out only the --config argument, this argument is used to
    # load a yaml file containing key-values that override the defaults for the main parser below
    parser_config = argparse.ArgumentParser(description='Training Yaml Config', add_help=False)
    parser_config.add_argument('-c', '--config', type=str, default='',
                        help='YAML config file specifying default arguments (default='')')

    # The main parser. Its default value will be overwritten by yaml config and 
    # the parsed value from command line will overwrite the corresponding value in yaml config.
    parser = argparse.ArgumentParser(description='Training', parents=[parser_config])
    # modelarts
    group = parser.add_argument_group('modelarts')
    group.add_argument('--enable_modelarts', type=bool, default=False,
                       help='Run on modelarts platform (default=False)')
    group.add_argument('--device_target', type=str, default='Ascend',
                       help='Target device, only used on modelarts platform (default=Ascend)')
    group.add_argument('--multi_data_url', type=str, default='/cache/data/',
                       help='path to multi dataset')
    group.add_argument('--data_url', type=str, default='/cache/data/',
                       help='path to dataset')
    group.add_argument('--ckpt_url', type=str, default='/cache/output/',
                       help='pre_train_model path in obs')
    group.add_argument('--train_url', type=str, default='/cache/output/',
                       help='model folder to save/load')

    # system
    group = parser.add_argument_group('system')
    group.add_argument('--distribute', type=_str2bool, nargs='?', const=True, default=False,
                       help='Distributed training mode (default=False)')
    group.add_argument('--mode', type=int, default=0,
                       help='Running in GRAPH_MODE(0) or PYNATIVE_MODE(1) (default=0)')
    group.add_argument('--val_while_train', type=_str2bool, nargs='?', const=True, default=True,
                       help='Velidate model performance on evaluation set while training (default=False)')
    group.add_argument('--amp_level', type=str, default='O0',
                       help='Amp level - Auto Mixed Precision level for saving memory and acceleration. '
                            'Choice: O0 - all FP32, O1 - only cast ops in white-list to FP16, '
                            'O2 - cast all ops except for blacklist to FP16, '
                            'O3 - cast all ops to FP16. (default="O0").')
    group.add_argument('--drop_overflow_update', type=_str2bool, nargs='?', const=True, default=False,
                       help='Whether to execute optimizer if there is an overflow (default=False)')
    group.add_argument('--seed', type=int, default=42,
                       help='Seed value for determining randomness in numpy, random, and mindspore (default=42)')

    # environment-related args in train/eval 
    group = parser.add_argument_group('train/eval data and checkpoint paths')
    group.add_argument('--dataset_root', type=str, default='./',
                       help='Path to root of the dataset (default="./")')
    group.add_argument('--ckpt_save_dir', type=str, default="./ckpt",
                       help='Dir path to save checkpoint (default="./ckpt")')
    group.add_argument('--ckpt_load_path', type=str, default='',
                       help='Load model weight from this checkpoint.'
                            'If resume training, specify the checkpoint path (default="")')
    # train loader
    group = parser.add_argument_group('train-loader')
    group.add_argument('--batch_size', type=int, default=8,
                       help='Batch size(default=8)')
    group.add_argument('--num_workers', type=int, default=8,
                       help='Number of parallel works for training data loader (default=8)')

    # Scheduler parameters
    group = parser.add_argument_group('Scheduler parameters')
    group.add_argument('--scheduler', type=str, default='cosine_decay',
                       choices=['constant', 'cosine_decay', 'exponential_decay', 'step_decay', 'multi_step_decay'],
                       help='Type of scheduler (default="cosine_decay")')
    group.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate (default=0.001)')
    group.add_argument('--min_lr', type=float, default=1e-6,
                       help='The minimum value of learning rate if scheduler supports (default=1e-6)')
    group.add_argument('--num_epochs', type=int, default=90,
                       help='Train epoch size (default=90)')
    group.add_argument('--warmup_epochs', type=int, default=3,
                       help='Warmup epochs (default=3)')
    group.add_argument('--decay_epochs', type=int, default=100,
                       help='Decay epochs (default=100)')


    return parser_config, parser


def parse_args():
    parser_config, parser = create_parser()
    args_config, remaining = parser_config.parse_known_args()
    
    # load yaml
    yaml_fp = args_config.config
    with open(yaml_fp, 'r') as fp:
        cfg = yaml.safe_load(fp)

    parser = _update_parser_default_values_by_config(parser, cfg)
    
    parser.set_defaults(config=args_config.config) # also add the yaml config arg to the final parser

    args = parser.parse_args(remaining) 

    return args 


# Note: when a new external arg is added, check whether the update rule fits. Add rule for it if needed. 
def _update_parser_default_values_by_config(parser, cfg_dict):
    ''' only update the default values that can be found in yaml file '''
    new_def_vals= {}
    args = parser.parse_args() # only to retrieve keys
    for k, v in args._get_kwargs():
        # rule tree
        if k in cfg_dict['system']:
            v = cfg_dict['system'][k] 
        elif k in cfg_dict['train']:
            v = cfg_dict['train'][k]
        elif k in cfg_dict['train']['dataset']:
            v = cfg_dict['train']['dataset'][k]
        elif k in cfg_dict['train']['loader']:
            v = cfg_dict['train']['loader'][k]
        elif k in cfg_dict['scheduler']:
            v = cfg_dict['scheduler'][k]

        new_def_vals[k] = v

    parser.set_defaults(**new_def_vals)

    return parser
    

# Note: when a new external args is added, check whether the merge rule fits and add rules if needed. 
def merge_args_to_config(args, cfg_dict):
    """
    Only merge the parsed args that can be found in the yaml file into the config dict.
    """
    for k, v in args._get_kwargs():
        # rule tree
        if k in cfg_dict['system']:
            cfg_dict['system'][k] = v
        elif k in cfg_dict['train']:
            cfg_dict['train'][k] = v
        elif k in cfg_dict['train']['dataset']:
            cfg_dict['train']['dataset'][k] = v
            if 'eval' in cfg_dict and k == 'dataset_root':
                cfg_dict['eval']['dataset'][k] = v
        elif k in cfg_dict['train']['loader']:
            cfg_dict['train']['loader'][k] = v
        elif k in cfg_dict['scheduler']:
            cfg_dict['scheduler'][k] = v
    
    return cfg_dict


def _str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "1"):
        return True
    elif v.lower() in ("no", "false", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")



if __name__ == '__main__':
    args = parse_args()
    print(vars(args).keys())
    for k, v in args._get_kwargs():
        print(k, v)
