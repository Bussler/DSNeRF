from turtle import position
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class FastNeRF(nn.Module):

    def __init__(self, D_pos = 8, D_view = 4, W_pos = 384, W_view = 128, input_ch = 3, input_ch_views = 3, output_ch = 3, output_ch_view = 1, skips_pos = [4], skips_view = [], use_viewdirs = True, cache_factor = 8, cache_resolution = 768):
        super(FastNeRF, self).__init__()
        
        self.D_pos = D_pos
        self.D_view = D_view

        self.W_pos = W_pos
        self.W_view = W_view

        self.input_ch = input_ch
        # D: Paper uses unit vectors as input for viewing direction
        self.input_ch_views = input_ch_views

        self.output_ch = output_ch
        self.output_ch_view = output_ch_view

        self.skips_pos = skips_pos
        self.skips_view = skips_view

        self.use_viewdirs = use_viewdirs

        # D: Number of cache components that are output by the network, paper has best results with 6-8 components
        self.cache_factor = cache_factor
        # D: dimensional resolution of cache, cubed
        self.cache_resolution = cache_resolution 

        # D: first network that takes position (x, y, z) as input and 
        # calculates F_pos by outputting (u_i, v_i, w_i) with 0 <= i < cache_factor
        
        self.position_mlp = nn.ModuleList([nn.Linear(input_ch, W_pos)] + [ nn.Linear(W_pos, W_pos) if i not in self.skips_pos else nn.Linear(W_pos + input_ch, W_pos) for i in range(D_pos-1)])
        self.position_final = nn.Linear(W_pos, output_ch * cache_factor)

        # D: second network that takes viewing direction (theta, phi) as input and
        # calculates F_dir by outputting beta_i with 0 <= i < cache_factor
        self.direction_mlp = nn.ModuleList([nn.Linear(input_ch_views, W_view)] + [ nn.Linear(W_view, W_view) for i in range(D_view-1)])
        self.view_final = nn.Linear(W_view, output_ch_view * cache_factor)

        self.alpha_linear = nn.Linear(W_pos, 1)


    def forward(self, x):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        h = input_pts


        for i, layer in enumerate(self.position_mlp):
            h = self.position_mlp[i](h)
            h = self.activation(h)
            if i in self.skips_pos:
                h = torch.cat([input_pts, h], -1)
        alpha = self.alpha_linear(h)
        
        position_features = self.position_final(h)

        # TODO: cache results
        position_features_reshaped = torch.reshape(position_features, shape=(x.shape[0], self.cache_factor, self.output_ch))
        h = input_views
        for i, layer in enumerate(self.direction_mlp):
            h = self.direction_mlp[i](h)
            h = self.activation(h)

        view_features = self.view_final(h)
        # TODO: cache results
        view_features_reshaped = torch.reshape(view_features, shape=(x.shape[0], self.cache_factor, self.output_ch_view))
        

        result = torch.sum(position_features_reshaped * view_features_reshaped, dim=1)
        return torch.cat([result, alpha], -1)



    def activation(self, x):
        return F.relu(x)


    

