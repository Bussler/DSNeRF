# M: Implementation of the NeRF network defined in run_nerf_helpers.py mixed with SIREN periodic activation functions https://www.vincentsitzmann.com/siren/

import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# M: taken from https://github.com/vsitzmann/siren
class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.
    
    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the 
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a 
    # hyperparameter.
    
    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of 
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)
    
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))
    

# Model
class SIRENNeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=False, omega_0 = 30):
        """ 
        """
        super(SIRENNeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        
        # M: 8 first input linear layers for 3D position x to predict volume density
        # M: Here, we intitialize them as SIREN Sine layers
        self.pts_linears = nn.ModuleList(
            [SineLayer(input_ch, W, is_first=True)] + [SineLayer(W, W) if i not in self.skips else SineLayer(W + input_ch, W) for i in range(D-1)])
        
        # M: last linear layer for output of prev 3D position (vol density) and view dir to predict view-dependent color
        # M: Here, we intitialize them as SIREN Sine layers
        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([SineLayer(input_ch_views + W, W//2)])

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])
        
        # M: Final output layers to generate rgb a values in needed dimensions from prev layers

        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W//2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, x):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            # M: Here we already use sin activation function in SineLayer
            #h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)
        
            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                # M: Here we already use sin activation function in SineLayer
                #h = F.relu(h)

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        return outputs    

    def load_weights_from_keras(self, weights):
        assert self.use_viewdirs, "Not implemented if use_viewdirs=False"
        
        # Load pts_linears
        for i in range(self.D):
            idx_pts_linears = 2 * i
            self.pts_linears[i].weight.data = torch.from_numpy(np.transpose(weights[idx_pts_linears]))    
            self.pts_linears[i].bias.data = torch.from_numpy(np.transpose(weights[idx_pts_linears+1]))
        
        # Load feature_linear
        idx_feature_linear = 2 * self.D
        self.feature_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_feature_linear]))
        self.feature_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_feature_linear+1]))

        # Load views_linears
        idx_views_linears = 2 * self.D + 2
        self.views_linears[0].weight.data = torch.from_numpy(np.transpose(weights[idx_views_linears]))
        self.views_linears[0].bias.data = torch.from_numpy(np.transpose(weights[idx_views_linears+1]))

        # Load rgb_linear
        idx_rbg_linear = 2 * self.D + 4
        self.rgb_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear]))
        self.rgb_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear+1]))

        # Load alpha_linear
        idx_alpha_linear = 2 * self.D + 6
        self.alpha_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear]))
        self.alpha_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear+1]))



# Model
class SIRENNeRF2(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=False, omega_0 = 30):
        """ 
        """
        super(SIRENNeRF2, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        
        # M: 8 first input linear layers for 3D position x to predict volume density
        # M: Here, we intitialize them as SIREN Sine layers
        self.linearSigma = nn.ModuleList(
            [SineLayer(input_ch, W, is_first=True)] + [SineLayer(W, W) if i not in self.skips else SineLayer(W + input_ch, W) for i in range(D-1)])

        self.outputSigma = nn.Linear(W, 1)
        
        # M: last linear layer for output of prev 3D position (vol density) and view dir to predict view-dependent color
        # M: Here, we intitialize them as SIREN Sine layers
        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.linearRGB = nn.ModuleList(
            [SineLayer(input_ch, W, is_first=True)] + [SineLayer(W, W) if i not in self.skips else SineLayer(W + input_ch, W) for i in range(D-1)])

        self.intermediateRGB = nn.Linear(W, W)
        self.intermediateRGBSIREN = SineLayer(W, W)
        self.outputRGB = nn.Linear(W, 3)


    # Only xyz are used here
    def forward(self, x):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
       
        # sigma
        h = input_pts
        for i, l in enumerate(self.linearSigma):
            h = self.linearSigma[i](h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        sigma = F.relu(self.outputSigma(h))

        # rgb
        h = input_pts
        for i, l in enumerate(self.linearRGB):
            h = self.linearRGB[i](h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        h = self.intermediateRGB(h)
        h = self.intermediateRGBSIREN(h)

        rgb = F.sigmoid(self.outputRGB(h))

        outputs = torch.cat([rgb, sigma], -1)

        return outputs    
