import torch
from torchvision.transforms import ToTensor
from timm.models import create_model
from timm.models.helpers import checkpoint_seq

from PIL import Image
import os

def unpatchify(x: torch.Tensor, patch_size) -> torch.Tensor:
        """Combine non-overlapped patches into images.
        Args:
            x (torch.Tensor): The shape is (N, L, patch_size**2 *3)
        Returns:
            imgs (torch.Tensor): The shape is (N, 3, H, W)
        """
        p = patch_size
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs


class CDModel(torch.nn.Module):
  """Model that takes two images as input and produces one as output.
  It uses a transformer without positional encoding to encode the information of one image.
  The encoding is given along with the other image to another transformer that will produce the output image.
  """
  def __init__(self):
    super().__init__()
    self.vit_enc = create_model('vit_base_patch32_224', pretrained=True)
    # remove positional encoding
    with torch.no_grad():
      self.vit_enc.pos_embed = torch.nn.Parameter(torch.zeros_like(self.vit_enc.pos_embed), requires_grad=False)  # should be zero and not trained

    self.vit_dec = create_model('vit_base_patch16_224', pretrained=False)
    self.vit_dec.cls_token = None  # we'll replace it by hacking

  def forward(self, img1, img2):
    z = self.vit_enc.forward_head(self.vit_enc.forward_features(img2), pre_logits=True)[:, None, :]  # (B, 1, F)
    x = self.vit_dec.patch_embed(img1)
    assert not self.vit_dec.no_embed_class, "needed for the hack to work"
    x = torch.cat((z, x), dim=1)  # (B, L=14x14+1, F=3x16x16)
    x = self.vit_dec._pos_embed(x)
    x = self.vit_dec.norm_pre(x)
    if self.vit_dec.grad_checkpointing and not torch.jit.is_scripting():
        x = checkpoint_seq(self.vit_dec.blocks, x)
    else:
        x = self.vit_dec.blocks(x)
    x = self.vit_dec.norm(x)[:, 1:]
    pred_diff = unpatchify(x, patch_size=16)  # (B, 3, 224, 224)
    return pred_diff + img1  # residual connection






