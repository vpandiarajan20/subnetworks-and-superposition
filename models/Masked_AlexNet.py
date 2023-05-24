import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import numpy as np
from torchvision.models import AlexNet
from functools import partial

### using SoftMaskedConv2d originally from the continuous sparsification 
### paper, but much modified to make easier to use

def sigmoid(x):
    return float(1./(1.+np.exp(-x)))

class SoftMaskedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=1, stride=1, mask_initial_value=0., temp=1, ticket=False):
        super(SoftMaskedConv2d, self).__init__()
        self.mask_initial_value = mask_initial_value
        
        self.in_channels = in_channels
        self.out_channels = out_channels    
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.temp = temp
        self.ticket = ticket
        
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        nn.init.xavier_normal_(self.weight)
        
    def init_mask(self):
        self.weight_mask = nn.Parameter(torch.Tensor(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size))
        nn.init.constant_(self.weight_mask, 0)
        
        self.bias_mask = nn.Parameter(torch.Tensor(self.out_channels))
        nn.init.constant_(self.bias_mask, 0)


    def compute_mask(self):
        #scaling = 1. / sigmoid(self.mask_initial_value)
        if self.ticket: mask = (self.weight_mask > 0).float() 
        else: mask = F.sigmoid(self.temp * self.weight_mask)
        return mask # * scaling   
    
    def compute_mask_bias(self):
        #scaling = 1. / sigmoid(self.mask_initial_value)
        if self.ticket: mask = (self.bias_mask > 0).float()
        else: mask = F.sigmoid(self.temp * self.bias_mask)
        return mask # * scaling
        
    def prune(self, temp):
        self.weight_mask.data = torch.clamp(temp * self.weight_mask.data, max=self.mask_initial_value)
        self.bias_mask.data = torch.clamp(temp * self.bias_mask.data, max=self.mask_initial_value)

    def forward(self, x):
        self.mask = self.compute_mask()
        masked_weight = self.weight * self.mask
    
        self.mask_bias = self.compute_mask_bias()
        masked_bias = self.bias * self.mask_bias
        out = F.conv2d(x, masked_weight, bias=masked_bias, stride=self.stride, padding=self.padding)        
        return out

    def extra_repr(self):
        return '{}, {}, kernel_size={}, stride={}, padding={}'.format(
            self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding)
 
class MaskedNet(nn.Module):
    def __init__(self):
        super(MaskedNet, self).__init__()
        self.ticket = False

    def checkpoint(self):
        for m in self.mask_modules: m.checkpoint()
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.Linear):
                m.checkpoint = copy.deepcopy(m.state_dict())

    def rewind_weights(self):
        for m in self.mask_modules: m.rewind_weights()
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.Linear):
                m.load_state_dict(m.checkpoint)
                
    def prune(self):
        for m in self.mask_modules: m.prune(self.temp)
            
class AlexNetMasking(MaskedNet):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.5, masked_layers=[0, 3, 6, 8, 10]) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.dropout = dropout
        Conv = partial(SoftMaskedConv2d, mask_initial_value=0.0) 

        layers = []

        if 0 in masked_layers:
            layers.append(Conv(3, 64, kernel_size=11, stride=4, padding=2))
        else:
            layers.append(nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2))

        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2))

        
        if 3 in masked_layers:
            layers.append(Conv(64, 192, kernel_size=5, padding=2))
        else:
            layers.append(nn.Conv2d(64, 192, kernel_size=5, padding=2),)

        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2))

        if 6 in masked_layers:
            layers.append(Conv(192, 384, kernel_size=3, padding=1))
        else:
            layers.append(nn.Conv2d(192, 384, kernel_size=3, padding=1))
        
        layers.append(nn.ReLU(inplace=True))
        
        if 8 in masked_layers:
            layers.append(Conv(384, 256, kernel_size=3, padding=1))
        else:
            layers.append(nn.Conv2d(384, 256, kernel_size=3, padding=1))
        
        layers.append(nn.ReLU(inplace=True))

        if 10 in masked_layers:
            layers.append(Conv(256, 256, kernel_size=3, padding=1))
        else:
            layers.append(nn.Conv2d(256, 256, kernel_size=3, padding=1))
        
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2))



        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
        self.mask_modules = [m for m in self.modules() if isinstance(m, SoftMaskedConv2d)]
        self.temp = 1
    
    def set_temp(self, new_temp):
        self.temp = new_temp
        for conv_mask in self.mask_modules:
            conv_mask.temp = self.temp
    
    def set_ticket(self, ticket):
        self.ticket = ticket
        for conv_mask in self.mask_modules:
            conv_mask.ticket = self.ticket

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    def model_init_mask(self):
        for p in self.parameters():
            p.requires_grad_(False)
        for child_sub in self.features:
            if isinstance(child_sub, SoftMaskedConv2d):
                child_sub.init_mask()
                child_sub.weight_mask.requires_grad_(True)
                child_sub.bias_mask.requires_grad_(True)

    def run_to_conv_layer(self, batch, layer_name_stop_point):
        features = list(self.named_children())[0][1]
        for layer_name, layer in list(features.named_children()):
            batch = layer(batch)
            if layer_name == layer_name_stop_point:
                break
        return batch



#### (might try to confirm continuous sparsification working right)

class L0Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1, dilation=1, groups=1, bias=False, l0=False, mask_init_value=0., temp: float = 1., ablate_mask=None):
        super(L0Conv2d, self).__init__()
        self.l0 = l0
        self.mask_init_value = mask_init_value
        
        self.in_channels = in_channels
        self.out_channels = out_channels    
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.dilation = _pair(dilation)
        self.groups = groups
        self.temp = temp
        self.ablate_mask=ablate_mask

        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        if self.ablate_mask == "random":
            self.random_weight = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))
            init.kaiming_uniform_(self.random_weight, a=math.sqrt(5))
            self.random_weight.requires_grad=False

        
    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(k), 1/sqrt(k)), where k = weight.size(1) * prod(*kernel_size)
        # For more details see: https://github.com/pytorch/pytorch/issues/15314#issuecomment-477448573
        if self.l0:
            self.init_mask()

        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                init.uniform_(self.bias, -bound, bound)
                
    def init_mask(self):
        self.mask_weight = nn.Parameter(torch.Tensor(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size))
        nn.init.constant_(self.mask_weight, self.mask_init_value)

    def compute_mask(self):
        if (self.ablate_mask == None) and (not self.training or self.mask_weight.requires_grad == False): 
            mask = (self.mask_weight > 0).float() # Hard cutoff once frozen or testing
        elif (self.ablate_mask != None) and (not self.training or self.mask_weight.requires_grad == False): 
            mask = (self.mask_weight <= 0).float() # Used for subnetwork ablation
        else:
            mask = F.sigmoid(self.temp * self.mask_weight)
        return mask 

    def train(self, train_bool):
        self.training = train_bool         
        
    def forward(self, x):
        if self.l0:
            self.mask = self.compute_mask()
            if self.ablate_mask == "random":
                masked_weight = self.weight * self.mask # This will give you the inverse weights, 0's for ablated weights
                masked_weight += (~self.mask.bool()).float() * self.random_weight # Invert the mask to target the remaining weights, make them random
            else:
                masked_weight = self.weight * self.mask
        else:
            masked_weight = self.weight
        
        out = F.conv2d(x, masked_weight, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)   
        return out


def extra_repr(self):
    return '{}, {}, kernel_size={}, stride={}, padding={}'.format(
        self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding)

class AlexNetMaskingL0Convs(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.5, 
                isL0: bool = False,
                mask_init_value: float = 0.,
                embed_dim:int = 10,
                ablate_mask:str = None) -> None:
        
        super().__init__()
        self.num_classes = num_classes
        self.dropout = dropout
        
        self.isL0 = isL0
        self.ablate_mask = ablate_mask # Used during testing to see performance when found mask is removed
        self.embed_dim = embed_dim
        self.temp = 1.
        
        L0_Conv = functools.partial(L0Conv2d, l0=True, mask_init_value=mask_init_value, ablate_mask=self.ablate_mask)
        Conv = functools.partial(L0Conv2d, l0=False)
        self.features = nn.Sequential(
            L0_Conv(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            L0_Conv(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            L0_Conv(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            L0_Conv(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            L0_Conv(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, L0Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                if m.weight is not None and m.bias is not None:
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                    
        self.mask_modules = [m for m in self.modules() if isinstance(m, SoftMaskedConv2d)]
        self.temp = 1
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    def model_init_mask(self):
        for p in self.parameters():
            p.requires_grad_(False)
        for child_sub in self.features:
            if isinstance(child_sub, L0_Conv):
                child_sub.init_mask()
                child_sub.weight_mask.requires_grad_(True)
                child_sub.bias_mask.requires_grad_(True)
                
    def get_temp(self):
        return self.temp

    def set_temp(self, temp):
        self.temp = temp
        for layer in self.modules():
            if type(layer) == L0Conv2d:
                layer.temp = temp


### helper functions for AlexNet
def l0_loss(self, model, y_hat, y, test_mode=False):
    if test_mode:
        error_loss = F.binary_cross_entropy_with_logits(y_hat, y, reduction="none")
    else:
        error_loss = F.binary_cross_entropy_with_logits(y_hat, y)

    l0_loss = 0.0
    masks = []
    for layer in model.modules():
        if hasattr(layer, "mask_weight"):
            masks.append(layer.mask)
    l0_loss = sum(m.sum() for m in masks)
    return (error_loss + (self.lamb * l0_loss), l0_loss)


def freeze_pretrained(model):
    for param in model.parameters():
        param.requires_grad = False