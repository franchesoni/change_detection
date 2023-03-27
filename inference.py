import shutil
import os
from PIL import Image
import numpy as np
import torch
import torchvision

from config import DATA_DIR
from data import CDDataModule
from lightning import CDLModule

def vist(img, name='img'):
  img = img[0].permute(1,2,0).detach().numpy()
  Image.fromarray((img*255).astype(np.uint8).squeeze()).save(name + '.png')


ckpt_dir = '/home/franchesoni/cdkllogs/version_2/checkpoints'
shutil.rmtree('tmp', ignore_errors=True)
shutil.copytree(ckpt_dir, 'tmp')
ckpt_path = os.listdir('tmp')[0]
model = CDLModule.load_from_checkpoint(os.path.join('tmp', ckpt_path))

###### FROM DATASET ######
datamodule = CDDataModule(data_dir=DATA_DIR, batch_size=1)
datamodule.setup()
dataloader = datamodule.train_dataloader()

for i, (img1, img2) in enumerate(dataloader):
    output = model.predict_step((img1, img2), i)
    vist(img1, 'img1')
    vist(img2, 'img2')
    vist(model.model(img1, img2), 'pred2')
    vist(output, 'diff')
    vist(output / output.max(), 'diffnorm')
    breakpoint()

##### FROM FOLDER WITH PNGs #####
datadir = 'vegas'
imgs = sorted(os.listdir(datadir))
# take combinations of 2 from imgs
for i in range(len(imgs)):
  for j in range(i+1, len(imgs)):
      print(imgs[i], imgs[j])
      img1 = Image.open(os.path.join(datadir, imgs[i])).resize((224, 224))
      img2 = Image.open(os.path.join(datadir, imgs[j])).resize((224, 224))
      img1 = torchvision.transforms.ToTensor()(img1)[None]
      img2 = torchvision.transforms.ToTensor()(img2)[None]
      output = model.predict_step((img1, img2), i)
      vist(img1, 'img1')
      vist(img2, 'img2')
      vist(model.model(img1, img2), 'pred2')
      vist(output, 'diff')
      vist(output / output.max(), 'diffnorm')
      breakpoint()

shutil.rmtree('tmp')
