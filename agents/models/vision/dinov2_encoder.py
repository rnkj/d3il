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


class DinoV2ViTcnn(nn.Module):
    def __init__(
        self,
        model_name: str,
        #patch_size: tuple[int, int] = (16, 16),
        hidden_size: int = 128,
        output_size: int = 32,
    ):
        super().__init__()

        #patch_h, patch_w = patch_size[:2]
        #print(patch_h, patch_w)
        inter_size = 10
        #inter_size = patch_h * patch_w * hidden_size
        print(inter_size)
        self.backbone = torch.hub.load("facebookresearch/dinov2", model_name)
        self.net = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            #nn.AdaptiveAvgPool2d((1, 1)),
            
        )
        self.fc = nn.Linear(256, 64)

    def forward(self, inputs: Tensor):
        with torch.no_grad():
            #print(inputs.shape)
            features_dict = self.backbone.forward_features(inputs)
            features = features_dict["x_norm_patchtokens"]
            
            features = features.permute(0, 2, 1)
            
            features = features.view(-1, 384, 16, 16)
            
        
        features = self.net(features)
        
        features = features.view(features.size(0), -1)
        
        features = self.fc(features)
        
        return features
        return self.fc(self.net(features).view(features.size(0), -1))
        return self.net(features)
    
class DinoV2ViTmean(nn.Module):
    def __init__(
        self,
        model_name: str,
        #patch_size: tuple[int, int] = (16, 16),
        hidden_size: int = 128,
        output_size: int = 32,
    ):
        super().__init__()

        #patch_h, patch_w = patch_size[:2]
        #print(patch_h, patch_w)
        inter_size = 10
        #inter_size = patch_h * patch_w * hidden_size
        print(inter_size)
        self.backbone = torch.hub.load("facebookresearch/dinov2", model_name)
        self.net = nn.Sequential(
            nn.Linear(self.backbone.num_features, hidden_size),
            nn.Flatten(),
            nn.Linear(inter_size, output_size),
        )

    def forward(self, inputs: Tensor):
        with torch.no_grad():
            #print(inputs.shape)
            features_dict = self.backbone.forward_features(inputs)
            features = features_dict["x_norm_patchtokens"]
            #print("########################")
            features = torch.mean(features, dim=1)
            #print(features.shape)
            #print("########################")
        return features
        