import pytorch_lightning as pl
from lightning import CDLModule
from data import CDDataModule
from config import DATA_DIR

def train():
  trainer = pl.Trainer(max_epochs=10, profiler='simple', fast_dev_run=True)
  module = CDLModule()
  datamodule = CDDataModule(data_dir=DATA_DIR)
  trainer.fit(module, datamodule=datamodule)

if __name__ == '__main__':
  train()