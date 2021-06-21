import math
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

def deform_butterfly_mult_torch(input, num_mat, R_parameters, R_shapes, return_intermediates=False):
    batch_size, n = input.shape[:2]
    output = input.contiguous()
    intermediates = [output]
    temp_p = 0
    for m in range(num_mat):
        R_shape = R_shapes[m]
        output_size, input_size, row, col, diag = R_shape[:]
        num_p = col * output_size
        t = R_parameters[temp_p:temp_p + num_p].view(input_size // (col * diag), diag, row, col).permute(0, 2, 3, 1)
        output_reshape = output.view(batch_size, input_size // (col * diag), 1, col, diag)
        output = (t * output_reshape).sum(dim=3)

        temp_p += num_p
        intermediates.append(output)
    
    return output.view(batch_size, output_size) if not return_intermediates else intermediates

class DeBut_2dConv(nn.Module):
    """Product of log N butterfly factors, each is a block 2x2 of diagonal matrices.
    Compatible with torch.nn.Linear. """

    def __init__(self, input_channel, output_channel, kernel_size, stride = 1, padding = 0, dilation = 1,
                R_shapes = [[]],
                bias=True, complex=False, return_intermediates = False):
        super().__init__()
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.padding = (padding, padding) if isinstance(padding, int) else padding
        self.dilation = (dilation, dilation) if isinstance(dilation, int) else dilation

        self.input_channel = input_channel
        self.output_channel = output_channel
        self.in_size = self.input_channel * self.kernel_size[0] * self.kernel_size[1]
        self.out_size = self.output_channel

        self.return_intermediates = return_intermediates
        
        # new parameters
        self.num_mat = len(R_shapes)
        self.R_shapes = R_shapes
        
        R_shapes_np = np.array(R_shapes)
        num_parameters = np.sum(R_shapes_np[:,0]*R_shapes_np[:,3])
        scaling = 1.0 / math.sqrt(2)
        self.twiddle = nn.Parameter(torch.randn((num_parameters)) * scaling)
        if bias:
            bias_shape = (self.out_size, ) 
            self.bias = nn.Parameter(torch.Tensor(*bias_shape))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize bias the same way as torch.nn.Linear."""
        if self.bias is not None:
            bound = 1 / math.sqrt(self.in_size)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        """
        Parameters:
            input: (batch, *, in_size) if real or (batch, *, in_size, 2) if complex
        Return:
            output: (batch, *, out_size) if real or (batch, *, out_size, 2) if complex
        """
        output = self.pre_process(input)
        output = deform_butterfly_mult_torch(output, self.num_mat, self.twiddle, self.R_shapes, self.return_intermediates)
        return self.post_process(input, output)

    def pre_process(self, input):
        batch, c, h, w = input.shape
        h_out = (h + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) // self.stride[0] + 1
        w_out = (w + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) // self.stride[1] + 1
        
        input_patches = F.unfold(input, self.kernel_size, self.dilation, self.padding, self.stride).view(
            batch, c, self.kernel_size[0] * self.kernel_size[1], h_out * w_out)
        output = input_patches.permute(0, 3, 2, 1).reshape(batch * h_out * w_out, self.kernel_size[0] * self.kernel_size[1] * c)
        return output
   
    def post_process(self, input, output):
        batch, c, h, w = input.shape
        h_out = (h + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) // self.stride[0] + 1
        w_out = (w + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) // self.stride[1] + 1
        if self.bias is not None:
            output += self.bias

        return output.view(batch, h_out * w_out, self.output_channel).transpose(1, 2).view(batch, self.output_channel, h_out, w_out)