import torch
from torch import nn, Tensor


class DinoV2ViT(nn.Module):
    def __init__(
        self,
        model_name: str,
        patch_size: tuple[int, int] = (16, 16),
        hidden_size: int = 128,
        output_size: int = 32,
    ):
        super().__init__()

        patch_h, patch_w = patch_size[:2]
        inter_size = patch_h * patch_w * hidden_size

        self.backbone = torch.hub.load("facebookresearch/dinov2", model_name)
        self.net = nn.Sequential(
            nn.Linear(self.backbone.num_features, hidden_size),
            nn.Flatten(),
            nn.Linear(inter_size, output_size),
        )

    def forward(self, inputs: Tensor):
        with torch.no_grad():
            features_dict = self.backbone.forward_features(inputs)
            features = features_dict["x_norm_patchtokens"]
        return self.net(features)
