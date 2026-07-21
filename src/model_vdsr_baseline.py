import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        
        # affine=True allows the network to learn its own custom scale/shift per channel
        self.norm = nn.InstanceNorm2d(out_channels, affine=True) 
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # Pass through Conv -> then Stabilize -> then Activate
        return self.relu(self.norm(self.conv(x)))


class GeomorphologicalLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0, gamma=2.0):
        """
        alpha: Weight for 0th Derivative (Absolute Elevation)
        beta:  Weight for 1st Derivative (Surface Incline / Slope)
        gamma: Weight for 2nd Derivative (Profile Curvature / Ridge Sharpness)
        """
        super(GeomorphologicalLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.l1 = nn.L1Loss()

        # 1. Sobel X (Horizontal Slope normalized to true unit gradient)
        sobel_x = torch.tensor([[ -1.,  0.,  1.],
                                [ -2.,  0.,  2.],
                                [ -1.,  0.,  1.]], dtype=torch.float32).view(1, 1, 3, 3) / 8.0

        # 2. Sobel Y (Vertical Slope normalized to true unit gradient)
        sobel_y = torch.tensor([[ -1., -2., -1.],
                                [  0.,  0.,  0.],
                                [  1.,  2.,  1.]], dtype=torch.float32).view(1, 1, 3, 3) / 8.0

        # 3. Laplacian (Curvature / Ridge & Valley detector)
        laplacian = torch.tensor([[  0.,  1.,  0.],
                                  [  1., -4.,  1.],
                                  [  0.,  1.,  0.]], dtype=torch.float32).view(1, 1, 3, 3)

        # register_buffer locks these kernels into the module state. 
        # When you send this loss to .to(device), these automatically ride along to the GPU.
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
        self.register_buffer('laplacian', laplacian)

    def forward(self, pred, gt):
        # 0th Derivative: Base Height
        base_loss = self.l1(pred, gt)

        # 1st Derivative: Slope magnitude
        pred_dx = F.conv2d(pred, self.sobel_x, padding=1)
        pred_dy = F.conv2d(pred, self.sobel_y, padding=1)
        gt_dx = F.conv2d(gt, self.sobel_x, padding=1)
        gt_dy = F.conv2d(gt, self.sobel_y, padding=1)
        
        slope_loss = self.l1(pred_dx, gt_dx) + self.l1(pred_dy, gt_dy)

        # 2nd Derivative: Curvature (Forces the lazy flat spots to spike up)
        pred_curve = F.conv2d(pred, self.laplacian, padding=1,mode='replicate')
        gt_curve = F.conv2d(gt, self.laplacian, padding=1)
        
        curve_loss = self.l1(pred_curve, gt_curve)

        return (self.alpha * base_loss) + (self.beta * slope_loss) + (self.gamma * curve_loss)

class BaselineVDSR(nn.Module):
    def __init__(self, num_layers=20, num_features=64):
        super(BaselineVDSR, self).__init__()
        
        layers = []
        # Input Layer: Takes 1-channel Low-Res DEM (C1)
        layers.append(ConvBlock(in_channels=1, out_channels=num_features))
        
        # Intermediate Deep Layers
        for _ in range(num_layers - 2):
            layers.append(ConvBlock(in_channels=num_features, out_channels=num_features))
            
        # Final Reconstruction Layer: Outputs 1-channel Residual Map
        self.deep_feature_extraction = nn.Sequential(*layers)
        
        self.reconstruction_layer = nn.Conv2d(
            in_channels=num_features,
            out_channels=1,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        
        # Weight Initialization using He (Kaiming) Normal method optimized for ReLU
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        # x shape: [Batch, 1, H, W] (Low-Resolution DEM scaled up via interpolation)
        residual = self.deep_feature_extraction(x)
        residual = self.reconstruction_layer(residual)
        
        # Global Skip Connection: Add the predicted details back to the base geometry
        out = x + residual
        return out

# =============================================================================
# Jupyter Cell Execution Verification
# =============================================================================
# Put this at the very bottom of model.py:

if __name__ == "__main__":
    sample_input = torch.randn(4, 1, 128, 128)
    model = BaselineVDSR(num_layers=20, num_features=64)
    with torch.no_grad():
        sample_output = model(sample_input)
        
    print("✅ Architecture loaded and forward pass successful.")
    print(f"   Input Tensor Shape:  {sample_input.shape}")
    print(f"   Output Tensor Shape: {sample_output.shape}")