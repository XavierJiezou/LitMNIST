import os
import torch
import pytorch_lightning as pl


data_config = {
    'data_dir': os.getcwd(),
    'batch_size': 256,
    'num_workers': 8,
    'persistent_workers': torch.cuda.device_count() > 0,
    'pin_memory': torch.cuda.device_count() > 0
}


model_config = {
    'channels': 1,
    'width': 28,
    'height': 28,
    'hidden_size': 64,
    'num_classes': 10
}


lighting_config = {
    'debug': False,
    'example_dims': (100, *list(model_config.values())[:3]),
    'learning_rate': 1e-3
}


train_config = {
    'log_every_n_steps': 0,
    'auto_scale_batch_size': None, # Batch size finder is not yet supported for DDP
    'auto_lr_find': False, # LR finder is not yet supported for DDP and only works with models having a single optimizer.
    'fast_dev_run': lighting_config['debug'], 
    'max_epochs': 100,
    'gpus': torch.cuda.device_count(),
    'precision': 16,
    'callbacks': [
        pl.callbacks.EarlyStopping(
            monitor='val_loss',
            mode='min',
            patience=10
        ),
        pl.callbacks.ModelCheckpoint(
            filename='{epoch}-{val_loss:.4f}-{val_acc:.4f}',
            monitor='val_loss',
            save_top_k=3,
            mode='min'
        )
    ],
    'profiler': None
}


pl_config = {
    'data': data_config,
    'model': model_config,
    'lightning': lighting_config,
    'train': train_config
}
