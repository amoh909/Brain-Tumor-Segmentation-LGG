import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()

        ## Encoder
        # Encoder Block 1
        self.conv1_1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.bn1_1 = nn.BatchNorm2d(64)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn1_2 = nn.BatchNorm2d(64)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2)

        # Encoder Block 2
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(128)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn2_2 = nn.BatchNorm2d(128)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2)

        # Encoder Block 3
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3_1 = nn.BatchNorm2d(256)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn3_2 = nn.BatchNorm2d(256)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(2)

        # Encoder Block 4
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4_1 = nn.BatchNorm2d(512)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn4_2 = nn.BatchNorm2d(512)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(2)

        ## Bottleneck
        self.bottle_conv1 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.bottle_bn1 = nn.BatchNorm2d(1024)
        self.bottle_relu1 = nn.ReLU(inplace=True)
        self.bottle_conv2 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)
        self.bottle_bn2 = nn.BatchNorm2d(1024)
        self.bottle_relu2 = nn.ReLU(inplace=True)

        ## Decoder
        # Decoder Level 4
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec_conv4_1 = nn.Conv2d(1024, 512, kernel_size=3, padding=1) ## Skip 512
        self.dec_bn4_1 = nn.BatchNorm2d(512)
        self.dec_conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.dec_bn4_2 = nn.BatchNorm2d(512)

        # Decoder Level 3
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec_conv3_1 = nn.Conv2d(512, 256, kernel_size=3, padding=1) ## Skip 256
        self.dec_bn3_1 = nn.BatchNorm2d(256)
        self.dec_conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.dec_bn3_2 = nn.BatchNorm2d(256)

        # Decoder Level 2
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec_conv2_1 = nn.Conv2d(256, 128, kernel_size=3, padding=1) ## Skip 128
        self.dec_bn2_1 = nn.BatchNorm2d(128)
        self.dec_conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.dec_bn2_2 = nn.BatchNorm2d(128)

        # Decoder Level 1
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec_conv1_1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)  ## Skip 64
        self.dec_bn1_1 = nn.BatchNorm2d(64)
        self.dec_conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.dec_bn1_2 = nn.BatchNorm2d(64)

        # Final 1x1 Convolution
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.relu1_1(self.bn1_1(self.conv1_1(x)))
        e1 = self.relu1_2(self.bn1_2(self.conv1_2(e1)))
        p1 = self.pool1(e1) 

        e2 = self.relu2_1(self.bn2_1(self.conv2_1(p1)))
        e2 = self.relu2_2(self.bn2_2(self.conv2_2(e2)))
        p2 = self.pool2(e2) 

        e3 = self.relu3_1(self.bn3_1(self.conv3_1(p2)))
        e3 = self.relu3_2(self.bn3_2(self.conv3_2(e3)))
        p3 = self.pool3(e3) 

        e4 = self.relu4_1(self.bn4_1(self.conv4_1(p3)))
        e4 = self.relu4_2(self.bn4_2(self.conv4_2(e4)))
        p4 = self.pool4(e4) 

        # Bottleneck
        b = self.bottle_relu1(self.bottle_bn1(self.bottle_conv1(p4)))
        b = self.bottle_relu2(self.bottle_bn2(self.bottle_conv2(b)))

        # Decoder
        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = torch.relu(self.dec_bn4_1(self.dec_conv4_1(d4)))
        d4 = torch.relu(self.dec_bn4_2(self.dec_conv4_2(d4)))

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = torch.relu(self.dec_bn3_1(self.dec_conv3_1(d3)))
        d3 = torch.relu(self.dec_bn3_2(self.dec_conv3_2(d3)))

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = torch.relu(self.dec_bn2_1(self.dec_conv2_1(d2)))
        d2 = torch.relu(self.dec_bn2_2(self.dec_conv2_2(d2)))
  
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = torch.relu(self.dec_bn1_1(self.dec_conv1_1(d1)))
        d1 = torch.relu(self.dec_bn1_2(self.dec_conv1_2(d1)))

        # Final Output
        out = self.final_conv(d1)
        return out

class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        """
        F_g: number of channels for gating signal (from lower, coarser scale)
        F_l: number of channels for the skip connection (from encoder)
        F_int: number of intermediate channels
        """
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class AttentionUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()

        ## Encoder
        # Encoder Block 1
        self.conv1_1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.bn1_1 = nn.BatchNorm2d(64)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn1_2 = nn.BatchNorm2d(64)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2)

        # Encoder Block 2
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(128)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn2_2 = nn.BatchNorm2d(128)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2)

        # Encoder Block 3
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3_1 = nn.BatchNorm2d(256)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn3_2 = nn.BatchNorm2d(256)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(2)

        # Encoder Block 4
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4_1 = nn.BatchNorm2d(512)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn4_2 = nn.BatchNorm2d(512)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(2)

        ## Bottleneck
        self.bottle_conv1 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.bottle_bn1 = nn.BatchNorm2d(1024)
        self.bottle_relu1 = nn.ReLU(inplace=True)
        self.bottle_conv2 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)
        self.bottle_bn2 = nn.BatchNorm2d(1024)
        self.bottle_relu2 = nn.ReLU(inplace=True)

        ## Decoder
        # Decoder Level 4
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.att4 = AttentionBlock(F_g=512, F_l=512, F_int=256)
        self.dec_conv4_1 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.dec_bn4_1 = nn.BatchNorm2d(512)
        self.dec_conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.dec_bn4_2 = nn.BatchNorm2d(512)

        # Decoder Level 3
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.att3 = AttentionBlock(F_g=256, F_l=256, F_int=128)
        self.dec_conv3_1 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.dec_bn3_1 = nn.BatchNorm2d(256)
        self.dec_conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.dec_bn3_2 = nn.BatchNorm2d(256)

        # Decoder Level 2
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.att2 = AttentionBlock(F_g=128, F_l=128, F_int=64)
        self.dec_conv2_1 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.dec_bn2_1 = nn.BatchNorm2d(128)
        self.dec_conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.dec_bn2_2 = nn.BatchNorm2d(128)

        # Decoder Level 1
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.att1 = AttentionBlock(F_g=64, F_l=64, F_int=32)
        self.dec_conv1_1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.dec_bn1_1 = nn.BatchNorm2d(64)
        self.dec_conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.dec_bn1_2 = nn.BatchNorm2d(64)

        # Final 1x1 Convolution
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.relu1_1(self.bn1_1(self.conv1_1(x)))
        e1 = self.relu1_2(self.bn1_2(self.conv1_2(e1)))
        p1 = self.pool1(e1) 

        e2 = self.relu2_1(self.bn2_1(self.conv2_1(p1)))
        e2 = self.relu2_2(self.bn2_2(self.conv2_2(e2)))
        p2 = self.pool2(e2) 

        e3 = self.relu3_1(self.bn3_1(self.conv3_1(p2)))
        e3 = self.relu3_2(self.bn3_2(self.conv3_2(e3)))
        p3 = self.pool3(e3) 

        e4 = self.relu4_1(self.bn4_1(self.conv4_1(p3)))
        e4 = self.relu4_2(self.bn4_2(self.conv4_2(e4)))
        p4 = self.pool4(e4) 

        # Bottleneck
        b = self.bottle_relu1(self.bottle_bn1(self.bottle_conv1(p4)))
        b = self.bottle_relu2(self.bottle_bn2(self.bottle_conv2(b)))

        # Decoder
        d4 = self.up4(b)
        x4 = self.att4(g=d4, x=e4)
        d4 = torch.cat([d4, x4], dim=1)
        d4 = torch.relu(self.dec_bn4_1(self.dec_conv4_1(d4)))
        d4 = torch.relu(self.dec_bn4_2(self.dec_conv4_2(d4)))

        d3 = self.up3(d4)
        x3 = self.att3(g=d3, x=e3)
        d3 = torch.cat([d3, x3], dim=1)
        d3 = torch.relu(self.dec_bn3_1(self.dec_conv3_1(d3)))
        d3 = torch.relu(self.dec_bn3_2(self.dec_conv3_2(d3)))

        d2 = self.up2(d3)
        x2 = self.att2(g=d2, x=e2)
        d2 = torch.cat([d2, x2], dim=1)
        d2 = torch.relu(self.dec_bn2_1(self.dec_conv2_1(d2)))
        d2 = torch.relu(self.dec_bn2_2(self.dec_conv2_2(d2)))
  
        d1 = self.up1(d2)
        x1 = self.att1(g=d1, x=e1)
        d1 = torch.cat([d1, x1], dim=1)
        d1 = torch.relu(self.dec_bn1_1(self.dec_conv1_1(d1)))
        d1 = torch.relu(self.dec_bn1_2(self.dec_conv1_2(d1)))

        # Final Output
        out = self.final_conv(d1)
        return out

def get_model(model_type="unet", in_channels=1, out_channels=1):
    model_type = model_type.lower()
    if model_type == "unet":
        return UNet(in_channels=in_channels, out_channels=out_channels)
    elif model_type == "attention_unet":
        return AttentionUNet(in_channels=in_channels, out_channels=out_channels)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Options are 'unet' or 'attention_unet'")