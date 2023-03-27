
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
    loss = torch.nn.functional.l1_loss(img2pred, img2) + torch.nn.functional.l1_loss(img1pred, img1)
    self.log('train_loss', loss, prog_bar=True)
    return loss

  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5)
    return {"optimizer":optimizer, "lr_scheduler":scheduler, "monitor":"train_loss"}


  def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
    img1, img2 = batch
    img2pred = self((img1, img2))
    img1pred = self((img2, img1))
    # take absolute differences
    img2diff = torch.abs(img2pred - img2).mean(dim=1, keepdim=True)
    img1diff = torch.abs(img1pred - img1).mean(dim=1, keepdim=True)
    # a change should be in both differences
    globaldiff = img2diff * img1diff
    return globaldiff

    


