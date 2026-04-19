import argparse
import inspect
import ast

from datasets import DataInterface
from models import ModelInterface
from utils.utils import *

# pytorch_lightning
import pytorch_lightning as pl
from pytorch_lightning import Trainer

#--->Setting parameters
def make_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', default='train', type=str)
    parser.add_argument('--config', default='Camelyon/TransMIL.yaml',type=str)
    parser.add_argument('--gpus', default='0', type=str,
                        help='GPU indices as a comma-separated string. Use "cpu" to force CPU mode.')
    parser.add_argument('--fold', default=0, type=int)
    args = parser.parse_args()
    return args


def parse_gpus(raw_gpus):
    if raw_gpus is None:
        return []

    if isinstance(raw_gpus, (list, tuple)):
        return [int(gpu) for gpu in raw_gpus]

    raw_gpus = str(raw_gpus).strip()
    if raw_gpus.lower() in {'cpu', 'none', ''}:
        return []

    if raw_gpus.startswith('['):
        return [int(gpu) for gpu in ast.literal_eval(raw_gpus)]

    return [int(gpu.strip()) for gpu in raw_gpus.split(',') if gpu.strip()]


def parse_major_minor(version_string):
    version_parts = []
    for chunk in str(version_string).split('.'):
        digits = ''.join(ch for ch in chunk if ch.isdigit())
        if not digits:
            break
        version_parts.append(int(digits))
        if len(version_parts) == 2:
            break

    while len(version_parts) < 2:
        version_parts.append(0)

    return tuple(version_parts[:2])


def resolve_precision(raw_precision, use_gpu):
    precision = int(raw_precision)
    if not use_gpu and precision == 16:
        return 32

    if precision != 16:
        return precision

    if parse_major_minor(pl.__version__) >= (2, 0):
        return '16-mixed' if use_gpu else 32

    return 16

#---->main
def main(cfg):

    #---->Initialize seed
    pl.seed_everything(cfg.General.seed)

    #---->load loggers
    cfg.load_loggers = load_loggers(cfg)

    #---->load callbacks
    cfg.callbacks = load_callbacks(cfg)

    #---->Define Data 
    DataInterface_dict = {'train_batch_size': cfg.Data.train_dataloader.batch_size,
                'train_num_workers': cfg.Data.train_dataloader.num_workers,
                'test_batch_size': cfg.Data.test_dataloader.batch_size,
                'test_num_workers': cfg.Data.test_dataloader.num_workers,
                'dataset_name': cfg.Data.dataset_name,
                'dataset_cfg': cfg.Data,}
    dm = DataInterface(**DataInterface_dict)

    #---->Define Model
    ModelInterface_dict = {'model': cfg.Model,
                            'loss': cfg.Loss,
                            'optimizer': cfg.Optimizer,
                            'data': cfg.Data,
                            'log': cfg.log_path
                            }
    model = ModelInterface(**ModelInterface_dict)
    
    #---->Instantiate Trainer
    trainer_kwargs = dict(
        num_sanity_val_steps=0, 
        logger=cfg.load_loggers,
        callbacks=cfg.callbacks,
        max_epochs= cfg.General.epochs,
        accumulate_grad_batches=cfg.General.grad_acc,
        deterministic=True,
        check_val_every_n_epoch=1,
    )
    trainer_signature = inspect.signature(Trainer.__init__)
    use_gpu = len(cfg.General.gpus) > 0

    if 'accelerator' in trainer_signature.parameters and 'devices' in trainer_signature.parameters:
        trainer_kwargs['accelerator'] = 'gpu' if use_gpu else 'cpu'
        trainer_kwargs['devices'] = len(cfg.General.gpus) if use_gpu else 1
        trainer_kwargs['precision'] = resolve_precision(cfg.General.precision, use_gpu)
        if use_gpu and len(cfg.General.gpus) > 1 and 'strategy' in trainer_signature.parameters:
            trainer_kwargs['strategy'] = cfg.General.multi_gpu_mode
    else:
        trainer_kwargs['gpus'] = cfg.General.gpus
        trainer_kwargs['amp_level'] = cfg.General.amp_level
        trainer_kwargs['precision'] = resolve_precision(cfg.General.precision, use_gpu)

    trainer = Trainer(**trainer_kwargs)

    #---->train or test
    if cfg.General.server == 'train':
        trainer.fit(model = model, datamodule = dm)
    else:
        model_paths = list(cfg.log_path.glob('*.ckpt'))
        model_paths = [str(model_path) for model_path in model_paths if 'epoch' in str(model_path)]
        for path in model_paths:
            print(path)
            new_model = model.load_from_checkpoint(
                checkpoint_path=path,
                data=cfg.Data,
                log=cfg.log_path,
            )
            trainer.test(model=new_model, datamodule=dm)

if __name__ == '__main__':

    args = make_parse()
    cfg = read_yaml(args.config)

    #---->update
    cfg.config = args.config
    cfg.General.gpus = parse_gpus(args.gpus)
    cfg.General.server = args.stage
    cfg.Data.fold = args.fold

    #---->main
    main(cfg)
 
