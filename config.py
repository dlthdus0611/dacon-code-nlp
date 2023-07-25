import argparse

def getConfig():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', default='Default', type=str, help='tag')
    parser.add_argument('--ver', default='unsup', type=str)
    parser.add_argument('--server', default='lsy_w', type=str)
    parser.add_argument('--exp_num', default='1', type=str)
    parser.add_argument('--experiment', default='Base', type=str)

    # model
    parser.add_argument('--lr', default=3e-5, type=float)
    parser.add_argument('--epochs', default=70, type=int)
    parser.add_argument('--max_len', default=512, type=int)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--initial_lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-3)

    # Path settings
    parser.add_argument('--fold', type=int, default=0, help='Validation Fold')
    parser.add_argument('--Kfold', type=int, default=5, help='Number of Split Folds')
    parser.add_argument('--model_path', type=str, default='results/')
    parser.add_argument('--file_name', default='df_0203', type=str)
    parser.add_argument('--ids_entity', default='ids_entity_0207', type=str)

    ## Scheduler
    parser.add_argument('--scheduler', type=str, default='cos')
    parser.add_argument('--warm_epoch', type=int, default=3)
    parser.add_argument('--freeze_epoch', type=int, default=0)
    ### Cosine Annealing
    parser.add_argument('--tmax', type=int, default = 145)
    parser.add_argument('--min_lr', type=float, default=5e-6)
    ### StepLR
    parser.add_argument('--milestone', type=int, nargs='*', default=[50])
    parser.add_argument('--lr_factor', type=float, default=0.1)
    parser.add_argument('--lr_step', type=int, default=5)

    ## etc.
    parser.add_argument('--type', type=str, default='diag')
    parser.add_argument('--patience', type=int, default=15, help='Early Stopping')
    parser.add_argument('--clipping', type=float, default=None, help='Gradient clipping')
    parser.add_argument('--re_training_exp', type=int, default=None)
    parser.add_argument('--use_weight_norm', type=bool, default=None, help='Weight Normalization')

    # Hardware settings
    parser.add_argument('--amp', default=True)
    parser.add_argument('--multi_gpu', type=bool, default=False)
    parser.add_argument('--logging', type=bool, default=False)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = getConfig()
    args = vars(args)
    print(args)
