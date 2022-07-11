import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class FastNeRF(nn.Module):
    def __init__(self, D_pos = 8, D_view = 4, W_pos = 384, W_view = 128, input_pos_ch = 3, input_view_ch = 3, skips_pos = [4], skips_view = [4], use_viewdirs = True, cache_factor = 8, cache_resolution = 768):
        super(FastNeRF, self).__init__()
        
        self.D_pos = D_pos
        self.D_view = D_view

        self.W_pos = W_pos
        self.W_view = W_view

        self.input_pos_ch = input_pos_ch
        # D: Paper uses unit vectors as input for viewing direction
        self.input_view_ch = input_view_ch

        self.skips_pos = skips_pos
        self.skips_view = skips_view

        self.use_viewdirs = use_viewdirs

        # D: Number of cache components that are output by the network, paper has best results with 6-8 components
        self.cache_factor = cache_factor
        # D: dimensional resolution of cache, cubed
        self.cache_resolution = cache_resolution 

        # D: first network that takes position (x, y, z) as input and 
        # calculates F_pos by outputting (u_i, v_i, w_i) with 0 <= i < cache_factor
        self.position_mlp = nn.Module()

        # D: second network that takes viewing direction (theta, phi) as input and
        # calculates F_dir by outputting beta_i with 0 <= i < cache_factor
        self.direction_mlp = nn.Module()


    

