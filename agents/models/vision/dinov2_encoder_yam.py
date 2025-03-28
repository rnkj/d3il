import torch
from torch import nn, Tensor




class SimpleCNN(nn.Module):
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

        print("SimpleCNN to test")
        self.stride=1

        self.backbone = torch.hub.load("facebookresearch/dinov2", model_name)
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=self.stride, padding=1),  # 16x16 -> 16x16
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 224x224 -> 112x112

            nn.Conv2d(32, 64, kernel_size=3, stride=self.stride, padding=1),  # 8x8 -> 8x8
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 112x112 -> 56x56

            nn.Conv2d(64, 64, kernel_size=3, stride=self.stride, padding=1),  # 8x8 -> 8x8
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 112x112 -> 56x56

            nn.Conv2d(64, 128, kernel_size=3, stride=self.stride, padding=1),  # 8x8 -> 8x8
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 56 -> 28

            nn.Flatten(),
            nn.Linear(196*128, 1000),
            nn.ReLU(),
            nn.Linear(1000, 64),
            nn.ReLU(),

        )

    def forward(self, inputs: Tensor):
        #with torch.no_grad():
            #features_dict = self.backbone.forward_features(inputs)
            #features = features_dict["x_norm_patchtokens"]
            #######features.shape is (batch_size, 384, 256)
        return self.net(inputs)
            
        return self.net(features)
    


class DinoV2ViTSimplefc(nn.Module):
    def __init__(
        self,
        model_name: str,
        patch_size: tuple[int, int] = (16, 16),
        hidden_size: int = 1000,
        output_size: int = 32,
    ):
        super().__init__()

        patch_h, patch_w = patch_size[:2]
        inter_size = patch_h * patch_w * hidden_size

        self.backbone = torch.hub.load("facebookresearch/dinov2", model_name)
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.backbone.num_features * patch_h * patch_w, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.ReLU(),
        )

    def forward(self, inputs: Tensor):
        with torch.no_grad():
            
            features_dict = self.backbone.forward_features(inputs)
            features = features_dict["x_norm_patchtokens"]
        return self.net(features)

