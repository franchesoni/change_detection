
import torch
import pytorch_lightning as pl
from model import CDModel

class CDLModule(pl.LightningModule):
  def __init__(self):
    super().__init__()
    self.model = CDModel()

  def forward(self, x):
    img1, img2 = x
    return self.model(img1, img2)

  def training_step(self, batch, batch_idx):
    img1, img2 = batch
    img2pred = self((img1, img2))
    img1pred = self((img2, img1))
    loss = torch.nn.functional.mse_loss(img2pred, img2) + torch.nn.functional.mse_loss(img1pred, img1)
    self.log('train_loss', loss, prog_bar=True)
    return loss

  def configure_optimizers(self):
      return torch.optim.Adam(self.parameters(), lr=0.001)

  def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
    img1, img2 = batch
    img2pred = self((img1, img2))
    img1pred = self((img2, img1))
    # take absolute differences
    img2diff = torch.abs(img2pred - img2)
    img1diff = torch.abs(img1pred - img1)
    # a change should be in both differences
    globaldiff = img2diff * img1diff
    return globaldiff

    


