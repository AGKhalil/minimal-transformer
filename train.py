import os
import random

import hydra
import numpy as np
import torch
from omegaconf import DictConfig
from omegaconf import OmegaConf
from torch import nn

import wandb
from minimal_transformer.datasets import MTDataset
from minimal_transformer.datasets import create_tokenized_dataset
from minimal_transformer.datasets import download_and_unzip_data
from minimal_transformer.models import TransformerModel
from minimal_transformer.trainers import Trainer


def set_seed(seed: int):
    """Setting one seed for different libraries (for reproducability)

    Args:
        seed (int): Fixed number to be used for reproducibility
            of results whenever the code runs
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@hydra.main(
    version_base=None,
    config_path="config",
    config_name="config",
)
def train(cfg: DictConfig) -> None:
    """Trains the model using the provided configuration.

    Args:
        cfg (DictConfig): Configuration parameters for training.

    Returns:
        None
    """
    if not os.path.exists(cfg.dataset.path):
        download_and_unzip_data(data_path=cfg.dataset.path)
        create_tokenized_dataset(
            data_path=cfg.dataset.path,
            dataset_size=cfg.dataset.dataset_size,
            sentence_length=cfg.dataset.sentence_length,
        )

    wandb.init(
        project=cfg.general.project,
        config=OmegaConf.to_container(cfg, resolve=[True | False]),
    )

    set_seed(cfg.general.seed)

    dataset = MTDataset(
        data_path=cfg.dataset.path,
        eng_csv_path=cfg.dataset.eng_csv_path,
        deu_csv_path=cfg.dataset.deu_csv_path,
    )
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset,
        cfg.dataset.train_test_split,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.dataset.batch_size,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=cfg.dataset.batch_size,
    )

    model = TransformerModel(
        enc_vocab_size=np.unique(dataset.source).shape[0] + 3,
        dec_vocab_size=np.unique(dataset.target).shape[0] + 3,
        **dict(
            cfg.model,
        ),
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.optimizer.learning_rate
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_loader=train_loader,
        test_loader=test_loader,
        logger=wandb,
        **dict(
            cfg.trainer,
        ),
    )
    trainer.fit()


if __name__ == "__main__":
    train()
