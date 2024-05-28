from data import OCT_Data
from arch_unet import N2NTrainer
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

if __name__ == "__main__":
    model = N2NTrainer(
        lr = 1e-4,
        )

    dm = OCT_Data(batch_size=16, workers=8)


    trainer = Trainer(
        callbacks = [
            LearningRateMonitor(logging_interval='step'),
            ModelCheckpoint(filename='{epoch}-{val_loss:.4f}', save_top_k=5, monitor='val_loss', mode='min'),
        ], 
        check_val_every_n_epoch=1,
        fast_dev_run=False,
        default_root_dir='checkpoint/nbr2nbr',

        deterministic=False, 
        max_epochs=50,

        devices = [0],
        # precision=16,
        # strategy='ddp',        
    )

    if False:
        trainer.test(model, dm)
    else:
        trainer.fit(model, dm)

