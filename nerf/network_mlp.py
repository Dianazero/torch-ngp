import torch
import torch.nn as nn
import torch.nn.functional as F

from encoding import get_encoder
from activation import trunc_exp
from .renderer import NeRFRenderer


class NeRFNetwork(NeRFRenderer):
    def __init__(self,
                 encoding="frequency",
                 encoding_dir="frequency",
                 num_layers=5,
                 hidden_dim=256,
                 geo_feat_dim=256,
                 num_layers_color=3,
                 hidden_dim_color=256,
                 bound=1,
                 **kwargs,
                 ):
        super().__init__(bound, **kwargs)

        # time network
        self.num_layers_time = 1
        # self.encoder_time, self.in_dim_time = get_encoder("frequency", input_dim = 1, multires = 8)
        self.out_dim_time = 256
        time_net = []
        for l in range(self.num_layers_time):
            if l == 0:
                in_dim = 1  #self.in_dim_time 
            else:
                in_dim = hidden_dim
            
            if l == self.num_layers_time - 1:
                out_dim = self.out_dim_time #self.in_dim_time
            else:
                out_dim = hidden_dim
            time_net.append(nn.Linear(in_dim, out_dim, bias=False))

        self.time_net = nn.ModuleList(time_net)



        # sigma network
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.geo_feat_dim = geo_feat_dim
        # self.encoder, self.in_dim = get_encoder("hashgrid", desired_resolution=2048 * bound)
        self.encoder, self.in_dim = get_encoder("frequency", input_dim = 3)

        sigma_net = []
        for l in range(num_layers):
            if l == 0:
                in_dim = self.in_dim + self.out_dim_time #self.in_dim_time 
            elif l == num_layers // 2:
                in_dim = self.hidden_dim  + self.in_dim
            else:
                in_dim = hidden_dim
            
            if l == num_layers - 1:
                out_dim = 1 + self.geo_feat_dim # 1 sigma + 15 SH features for color
            else:
                out_dim = hidden_dim
            
            sigma_net.append(nn.Linear(in_dim, out_dim, bias=False))

        self.sigma_net = nn.ModuleList(sigma_net)

        # color network
        self.num_layers_color = num_layers_color        
        self.hidden_dim_color = hidden_dim_color
        self.encoder_dir, self.in_dim_color = get_encoder(encoding_dir, multires = 8)
        self.in_dim_color += self.geo_feat_dim
        
        color_net =  []
        for l in range(num_layers_color):
            if l == 0:
                in_dim = self.in_dim_color # + self.in_dim_time 
            else:
                in_dim = hidden_dim_color
            
            if l == num_layers_color - 1:
                out_dim = 3 # 3 rgb
            else:
                out_dim = hidden_dim_color
            
            color_net.append(nn.Linear(in_dim, out_dim, bias=False))

        self.color_net = nn.ModuleList(color_net)
    
    def forward(self, x, d, t):
        # x: [N, 3], in [-bound, bound]
        # d: [N, 3], nomalized in [-1, 1]
        # t: [N, 1], num of frame sequence[0,nums_frames)
        # t = self.encoder_time(t)
        for l in range(self.num_layers_time):
            t = self.time_net[l](t)
            t = F.relu(t, inplace=True)

        # sigma
        x = self.encoder(x, bound=self.bound)
        #relu
        h = torch.cat([x, t], dim=-1)
        for l in range(self.num_layers):
            if l == self.num_layers // 2:
                h = torch.cat([h,x], dim = -1)
            h = self.sigma_net[l](h)
            if l != self.num_layers - 1:
                h = F.relu(h, inplace=True)

        #sigma = F.relu(h[..., 0])
        #exp activation
        sigma = trunc_exp(h[..., 0])
        geo_feat = h[..., 1:]

        # color
        
        d = self.encoder_dir(d)
        #relu
        h = torch.cat([d, geo_feat], dim=-1)
        h = torch.cat([h, t], dim=-1)
        for l in range(self.num_layers_color):
            h = self.color_net[l](h)
            if l != self.num_layers_color - 1:
                h = F.relu(h, inplace=True)
        
        # sigmoid activation for rgb
        color = torch.sigmoid(h)
        return sigma, color

    def density(self, x, t):
        # x: [N, 3], in [-bound, bound]
        # t: [N, 1], num of frame sequence[0,nums_frames)
        # t = self.encoder_time(t)
        for l in range(self.num_layers_time):
            t = self.time_net[l](t)
            t = F.relu(t, inplace=True)        

        
        x = self.encoder(x, bound=self.bound)

        h = torch.cat([x, t], dim=-1)
        for l in range(self.num_layers):
            if l == self.num_layers // 2:
                h = torch.cat([h,x], dim = -1)
            h = self.sigma_net[l](h)
            if l != self.num_layers - 1:
                h = F.relu(h, inplace=True)

        #sigma = F.relu(h[..., 0])
        sigma = trunc_exp(h[..., 0])
        geo_feat = h[..., 1:]

        return {
            'sigma': sigma,
            'geo_feat': geo_feat,
        }

    # allow masked inference
    def color(self, x, d, t, mask=None, geo_feat=None, **kwargs):
        # x: [N, 3] in [-bound, bound]
        # mask: [N,], bool, indicates where we actually needs to compute rgb.
        # t: [N, 1], num of frame sequence[0,nums_frames)

        if mask is not None:
            rgbs = torch.zeros(mask.shape[0], 3, dtype=x.dtype, device=x.device) # [N, 3]
            # in case of empty mask
            if not mask.any():
                return rgbs
            x = x[mask]
            d = d[mask]
            t = t[mask]
            geo_feat = geo_feat[mask]

        d = self.encoder_dir(d)
        # t = self.encoder_time(t)
        # for l in range(self.num_layers_time):
        #     t = self.time_net[l](t)
        #     t = F.relu(t, inplace=True)

        h = torch.cat([d, geo_feat], dim=-1)
        # h = torch.cat([h, t], dim=-1)
        for l in range(self.num_layers_color):
            h = self.color_net[l](h)
            if l != self.num_layers_color - 1:
                h = F.relu(h, inplace=True)
        
        # sigmoid activation for rgb
        h = torch.sigmoid(h)

        if mask is not None:
            rgbs[mask] = h.to(rgbs.dtype) # fp16 --> fp32
        else:
            rgbs = h

        return rgbs        
