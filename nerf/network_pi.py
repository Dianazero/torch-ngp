import torch
import torch.nn as nn
import torch.nn.functional as F

from encoding import get_encoder
from activation import trunc_exp
from .renderer import NeRFRenderer
def kaiming_leaky_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        torch.nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')

class FiLMLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.layer = nn.Linear(input_dim, hidden_dim)
        self.activate_layer = nn.LeakyReLU(0.2)

    def forward(self, x, freq, phase_shift):
        x = self.layer(x)
        freq = freq.expand_as(x)
        phase_shift = phase_shift.expand_as(x)
        return self.activate_layer(freq * x + phase_shift)

        # return torch.sin(freq * x + phase_shift)

class CustomMappingNetwork(nn.Module):
    def __init__(self, z_dim, map_hidden_dim, map_output_dim):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(z_dim, map_hidden_dim),
                                     nn.LeakyReLU(0.2, inplace=True),

                                    # nn.Linear(map_hidden_dim, map_hidden_dim),
                                    # nn.LeakyReLU(0.2, inplace=True),

                                    nn.Linear(map_hidden_dim, map_output_dim))

        self.network.apply(kaiming_leaky_init)
        with torch.no_grad():
            self.network[-1].weight *= 0.25

    def forward(self, z):
        frequencies_offsets = self.network(z)
        frequencies = frequencies_offsets[..., :frequencies_offsets.shape[-1]//2]
        phase_shifts = frequencies_offsets[..., frequencies_offsets.shape[-1]//2:]

        return frequencies, phase_shifts


class NeRFNetwork(NeRFRenderer):
    def __init__(self,
                 encoding="frequency",
                 encoding_dir="sphere_harmonics",
                 num_layers=4,
                 hidden_dim=128,
                 geo_feat_dim=15,
                 num_layers_color=2,
                 hidden_dim_color=128,
                 bound=1,
                 **kwargs,
                 ):
        super().__init__(bound, **kwargs)

        # time network

        self.encoder_time, self.in_dim_time = get_encoder("frequency", input_dim = 1, multires = 8)
        # freq and phase
        self.time_latent_encoder = CustomMappingNetwork(self.in_dim_time, hidden_dim, (num_layers + 1)*hidden_dim*2)

        # sigma network
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.geo_feat_dim = geo_feat_dim
        # self.encoder, self.in_dim = get_encoder("hashgrid", desired_resolution=2048 * bound)
        self.encoder, self.in_dim = get_encoder("frequency", input_dim = 3)

        sigma_net = []
        for l in range(num_layers):
            if l == 0:
                in_dim = self.in_dim  
            else:
                in_dim = hidden_dim
 
            out_dim = hidden_dim

            sigma_net.append(FiLMLayer(in_dim, out_dim))

        self.sigma_net = nn.ModuleList(sigma_net)

        self.final_layer_sigma = nn.Linear(hidden_dim, 1)
        self.final_layer_color = nn.Linear(hidden_dim, self.geo_feat_dim)
        

        # color network
        self.num_layers_color = num_layers_color        
        self.hidden_dim_color = hidden_dim_color
        self.encoder_dir, self.in_dim_color = get_encoder(encoding_dir)
        self.in_dim_color += self.geo_feat_dim
        
        self.color_layer_sine = FiLMLayer(self.in_dim_color, hidden_dim)
        self.color_layer_linear = nn.Sequential(nn.Linear(hidden_dim, 3), nn.Sigmoid())
        self.color_net = nn.ModuleList([self.color_layer_sine,self.color_layer_linear])

    def forward(self, x, d, t):
        # x: [N, 3], in [-bound, bound]
        # d: [N, 3], nomalized in [-1, 1]
        # t: [N, 1], num of frame sequence[0,nums_frames)
        t = self.encoder_time(t)
        frequencies, phase_shifts = self.time_latent_encoder(t)

        # sigma
        x = self.encoder(x, bound=self.bound)
        

        for index, layer in enumerate(self.sigma_net):
            start = index * self.hidden_dim
            end = (index+1) * self.hidden_dim
            x = layer(x, frequencies[..., start:end], phase_shifts[..., start:end])

        sigma = self.final_layer_sigma(x)
        geo_feat = self.final_layer_color(x)

        sigma = trunc_exp(sigma)
# 
        # color
        
        d = self.encoder_dir(d)
        #relu
        h = torch.cat([d, geo_feat], dim=-1)
    
        for index, layer in enumerate(self.color_net):
            if index != self.num_layers_color - 1:
                h = layer(h, frequencies[..., -self.hidden_dim:], phase_shifts[..., -self.hidden_dim:])
            else:
                h = layer(h)

        # sigmoid activation for rgb
        return sigma, color

    def density(self, x, t):
        # x: [N, 3], in [-bound, bound]
        # t: [N, 1], num of frame sequence[0,nums_frames)
        t = self.encoder_time(t)
        frequencies, phase_shifts = self.time_latent_encoder(t)

        # sigma
        x = self.encoder(x, bound=self.bound)
        

        for index, layer in enumerate(self.sigma_net):
            start = index * self.hidden_dim
            end = (index+1) * self.hidden_dim
            x = layer(x, frequencies[..., start:end], phase_shifts[..., start:end])

        sigma = self.final_layer_sigma(x)
        geo_feat = self.final_layer_color(x)

        sigma = trunc_exp(sigma)


        return {
            'sigma': sigma,
            'geo_feat': geo_feat,
        }

    # allow masked inference
    def color(self, x, d, t, mask=None, geo_feat=None, **kwargs):
        # x: [N, 3] in [-bound, bound]
        # mask: [N,], bool, indicates where we actually needs to compute rgb.
        # t: [N, 1], num of frame sequence[0,nums_frames)
        t = self.encoder_time(t)
        frequencies, phase_shifts = self.time_latent_encoder(t)
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
        #relu
        h = torch.cat([d, geo_feat], dim=-1)
        for index, layer in enumerate(self.color_net):
            if index != self.num_layers_color - 1:
                h = layer(h, frequencies[..., -self.hidden_dim:], phase_shifts[..., -self.hidden_dim:])
            else:
                h = layer(h)


        if mask is not None:
            rgbs[mask] = h.to(rgbs.dtype) # fp16 --> fp32
        else:
            rgbs = h

        return rgbs        
