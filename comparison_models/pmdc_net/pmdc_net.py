import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DepthwiseConvolutionBlock(nn.Module):
    
    def __init__(self, channels, kernel_size=3, padding=1):
        super().__init__()
        self.depthwise = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, padding=padding, 
                      groups=channels, bias=False),  
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.depthwise(x)

class PDCM(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, dilation: int = 2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.dw_d1 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1,
                      dilation=1, groups=out_channels, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.dw_d2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=2,
                      dilation=2, groups=out_channels, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.dw_d3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=3,
                      dilation=3, groups=out_channels, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        self.pointwise_after_assd = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        self.activation = nn.ReLU(inplace=True)

        self.fusion_1x1 = nn.Sequential(
            nn.Conv2d(in_channels + out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
      
        x1 = self.initial_conv(x)                    

        r1 = self.dw_d1(x1)
        r2 = self.dw_d2(x1)
        r3 = self.dw_d3(x1)
        r = r1 + r2 + r3

        x1_ass = self.activation(x1 + r)

        p = self.pointwise_after_assd(x1_ass)
        x2 = self.activation(x1 + p)

        y = self.fusion_1x1(torch.cat([x, x2], dim=1))
        return y

class BasicDoubleConv(nn.Module):
   
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class MSFAM(nn.Module):

    def __init__(self, in_channels, out_channels, reduction_ratio=4):
        super(MSFAM, self).__init__()
        
        mid_channels = max(in_channels // reduction_ratio, 16)
        
        self.dilated_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        
        self.dilated_conv2 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=3, dilation=3, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        
        self.dilated_conv3 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=5, dilation=5, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )

        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(mid_channels, mid_channels // 4, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels // 4, mid_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        self.depthwise_block = DepthwiseConvolutionBlock(mid_channels)

        self.fusion = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False),  
            nn.BatchNorm2d(out_channels)
        )

        self.residual_conv = nn.Identity() if in_channels == out_channels else nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        self.intermediate_activation = nn.ReLU(inplace=True)
        self.final_activation = nn.ReLU(inplace=True)
    
    def forward(self, x):

        identity = self.residual_conv(x)
        
        f1 = self.dilated_conv1(x)  
        f2 = self.dilated_conv2(x)  
        f3 = self.dilated_conv3(x) 
        
        fused = f1 + f2 + f3
        fused = self.intermediate_activation(fused)

        att_weights = self.attention(fused)
        attended = fused * att_weights

        enhanced = self.depthwise_block(attended)

        output = self.fusion(enhanced)

        final_output = identity + output
        
        return self.final_activation(final_output)


class ChannelWeightingModule(nn.Module):
    
    def __init__(self, channels, reduction_ratio=16, use_pdcm: bool = True):
     
        super(ChannelWeightingModule, self).__init__()
        self.channels = channels
           
        mid_channels = max(channels // reduction_ratio, 4)
        
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), 
            nn.Flatten(),             
            nn.Linear(channels, mid_channels, bias=False), 
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels, channels, bias=False),  
            nn.Sigmoid()             
        )

        self.double_conv = PDCM(channels, channels) if use_pdcm else BasicDoubleConv(channels, channels)
    
    def forward(self, low_feat, high_feat):

        low_weights = self.channel_attention(low_feat)  
        low_weights = low_weights.unsqueeze(2).unsqueeze(3)  
        
        high_weights = self.channel_attention(high_feat)  
        high_weights = high_weights.unsqueeze(2).unsqueeze(3)  
        
        weighted_low = low_feat * low_weights   
        weighted_high = high_feat * high_weights  
        
        fused_feat = weighted_low + weighted_high
        
        output = self.double_conv(fused_feat)
        
        return output


class DecoderAttentionFusionModule(nn.Module):

    def __init__(self, F_l, F_g, F_int):
        super().__init__()
        self.W_x = nn.Conv2d(F_l, F_int, kernel_size=1, bias=False)
        self.W_g = nn.Conv2d(F_g, F_int, kernel_size=1, bias=False)
        self.bias = nn.Parameter(torch.zeros(1, F_int, 1, 1))
        self.relu = nn.ReLU(inplace=True)
        self.psi = nn.Conv2d(F_int, 1, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, g):
        x_proj = self.W_x(x)
        g_proj = self.W_g(g)
        if g_proj.shape[2:] != x_proj.shape[2:]:
            g_proj = F.interpolate(g_proj, size=x_proj.shape[2:], mode='bilinear', align_corners=True)
        f = self.relu(x_proj + g_proj + self.bias)
        alpha = self.sigmoid(self.psi(f))

        return x * alpha + x


class PMDC_Net(nn.Module):
    def __init__(self, n_channels=1, n_classes=2, bilinear=False,
                 use_msfam=True, use_cwm=True, use_pdcm=True, use_dafm=True):
        super().__init__()
        factor = 2 if bilinear else 1

        self.use_pdcm = use_pdcm
        self.inc   = PDCM(n_channels, 64) if use_pdcm else BasicDoubleConv(n_channels, 64)
        self.msfam = MSFAM(64, 64) if use_msfam else nn.Identity()
        self.down1 = nn.Sequential(nn.MaxPool2d(2), PDCM(64, 128) if use_pdcm else BasicDoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), PDCM(128, 256) if use_pdcm else BasicDoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), PDCM(256, 512) if use_pdcm else BasicDoubleConv(256, 512))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), PDCM(512, 1024 // factor) if use_pdcm else BasicDoubleConv(512, 1024 // factor))

        self.use_dafm = use_dafm
        self.att4 = DecoderAttentionFusionModule(F_l=512, F_g=1024//factor, F_int=256) if use_dafm else nn.Identity()
        self.att3 = DecoderAttentionFusionModule(F_l=256, F_g=512//factor,  F_int=128) if use_dafm else nn.Identity()
        self.att2 = DecoderAttentionFusionModule(F_l=128, F_g=256//factor,  F_int=64) if use_dafm else nn.Identity()
        self.att1 = DecoderAttentionFusionModule(F_l=64,  F_g=128//factor,  F_int=32) if use_dafm else nn.Identity()

        self.up1 = self._make_up_layer_cwm(128, 64, bilinear)
        self.up2 = self._make_up_layer_cwm(256, 128//factor, bilinear)
        self.up3 = self._make_up_layer_cwm(512, 256//factor, bilinear)
        self.up4 = self._make_up_layer_cwm(1024, 512//factor, bilinear)
 
        self.use_cwm = use_cwm
        if use_cwm:
            self.cwm1 = ChannelWeightingModule(64, use_pdcm=use_pdcm)
            self.cwm2 = ChannelWeightingModule(128//factor, use_pdcm=use_pdcm)
            self.cwm3 = ChannelWeightingModule(256//factor, use_pdcm=use_pdcm)
            self.cwm4 = ChannelWeightingModule(512//factor, use_pdcm=use_pdcm)
        else:
            self.cwm1 = self.cwm2 = self.cwm3 = self.cwm4 = nn.Identity()
       
            self.dec_fuse4 = PDCM(1024, 512) if use_pdcm else BasicDoubleConv(1024, 512)
            self.dec_fuse3 = PDCM(512, 256)  if use_pdcm else BasicDoubleConv(512, 256)
            self.dec_fuse2 = PDCM(256, 128)  if use_pdcm else BasicDoubleConv(256, 128)
            self.dec_fuse1 = PDCM(128, 64)   if use_pdcm else BasicDoubleConv(128, 64)

        self.dropout = nn.Dropout2d(0.1)
        self.outc    = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
       
        x1 = self.inc(x)
        x1 = self.msfam(x1)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x4_att = self.att4(x4, x5) if self.use_dafm else x4
        x = self._upsample_with_cwm(self.up4, x5, x4_att, self.cwm4)

        x3_att = self.att3(x3, x) if self.use_dafm else x3
        x = self._upsample_with_cwm(self.up3, x, x3_att, self.cwm3)

        x2_att = self.att2(x2, x) if self.use_dafm else x2
        x = self._upsample_with_cwm(self.up2, x, x2_att, self.cwm2)

        x1_att = self.att1(x1, x) if self.use_dafm else x1
        x = self._upsample_with_cwm(self.up1, x, x1_att, self.cwm1)

        x = self.dropout(x)
        return self.outc(x)

    def _make_up_layer_cwm(self, in_ch, out_ch, bilinear):
        if bilinear:
            return nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            return nn.ConvTranspose2d(in_ch, in_ch//2, kernel_size=2, stride=2)

    def _upsample_with_cwm(self, up_layer, dec_feat, skip_feat, cwm_module):

        up = up_layer(dec_feat)
      
        diffY = skip_feat.size(2) - up.size(2)
        diffX = skip_feat.size(3) - up.size(3)
        up = F.pad(up, [diffX//2, diffX-diffX//2, diffY//2, diffY-diffY//2])
       
        if self.use_cwm:
            return cwm_module(skip_feat, up)
        else:
            x = torch.cat([skip_feat, up], dim=1)
            if up_layer is self.up4:
                return self.dec_fuse4(x)
            elif up_layer is self.up3:
                return self.dec_fuse3(x)
            elif up_layer is self.up2:
                return self.dec_fuse2(x)
            else:
                return self.dec_fuse1(x)

