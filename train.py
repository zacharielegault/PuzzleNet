import albumentations as A
import cv2
import torch
from albumentations.pytorch import ToTensorV2
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from nntools.dataset import Composition, ImageDataset, random_split
from nntools.utils import Config

from puzzlenet.callbacks import LogPredictionSamplesCallback
from puzzlenet.network import PuzzleNet

torch.set_float32_matmul_precision("high")


def main():
    seed_everything(1234, workers=True)
    config = Config("config.yaml")

    wandb_logger = WandbLogger(
        project="PuzzleNet",
        config={**config.tracked_params},
    )

    dataset = ImageDataset(
        "/home/tmp/clpla/data/imageNet/CLS-LOC/train/",
        recursive_loading=True,
        keep_size_ratio=True,
        flag=cv2.IMREAD_COLOR,
        shape=config["data"]["img_size"],
    )

    dataset.composer = Composition()
    dataset.composer << A.Normalize() << ToTensorV2()
    train_length = int(0.98 * len(dataset))
    val_length = len(dataset) - train_length
    train_set, val_set = random_split(dataset, [train_length, val_length])
    train_set.composer = train_set.composer
    val_set.composer = dataset.composer

    print(f"Training set size: {len(train_set)}, validation set size: {len(val_set)}")

    train_datloader = DataLoader(
        train_set,
        batch_size=config["data"]["batch_size"],
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=1,
        persistent_workers=True,
    )

    val_datloader = DataLoader(
        val_set,
        batch_size=config["data"]["batch_size"],
        shuffle=False,
        persistent_workers=True,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=1,
    )

    model = PuzzleNet(**config["model"])
    log_callback = LogPredictionSamplesCallback(wandb_logger=wandb_logger)

    model_best_ckpt = ModelCheckpoint(monitor="Validation loss", mode="min", auto_insert_metric_name=True)
    model_last_ckpt = ModelCheckpoint(save_last=True, auto_insert_metric_name=False)

    trainer = Trainer(
        # strategy="ddp_find_unused_parameters_false",
        logger=wandb_logger,
        callbacks=[log_callback, model_best_ckpt, model_last_ckpt],
        enable_checkpointing=True,
        **config["trainer"],
    )
    trainer.fit(model, train_dataloaders=train_datloader, val_dataloaders=val_datloader)


if __name__ == "__main__":
    main()
