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

class DeBut(nn.Module):
    """Product of log N butterfly factors, each is a block 2x2 of diagonal matrices.
    Compatible with torch.nn.Linear. """

    def __init__(self, in_size, out_size,
                R_shapes = [[48, 72, 2, 3, 1], [16, 48, 1, 3, 2], [16, 16, 2, 2, 4], [16, 16, 2, 2, 8]],
                bias=True, param='regular'):
        super().__init__()
        self.in_size = in_size
        m = int(math.ceil(math.log2(in_size)))
        self.m = m
        self.out_size = out_size
        self.param = param
        # new parameters
        self.num_mat = len(R_shapes)
        self.R_shapes = R_shapes
        
        if param == 'regular':
            R_shapes_np = np.array(R_shapes)
            num_parameters = np.sum(R_shapes_np[:,0]*R_shapes_np[:,3])
            scaling = 1.0 / math.sqrt(2)
            self.twiddle = nn.Parameter(torch.randn((num_parameters)) * scaling)
            print(self.twiddle.shape)
            # self.twiddle = self.gen_R_parameters()#nn.Parameter(torch.rand(R_parameters_shape) * scaling)
        # self.twiddle._is_structured = True  # Flag to avoid weight decay
        if bias:
            bias_shape = (out_size, ) 
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
            input: (batch, in_size) if real 
        Return:
            output: (batch, out_size) if real 
        """
        output = self.pre_process(input)
        if self.param == 'regular':
            output = deform_butterfly_mult_torch(output, self.num_mat, self.twiddle, self.R_shapes, return_intermediates=False)
        
        return self.deform_post_process(input, output) if self.param == 'regular' else self.post_process(input, output)

    def pre_process(self, input):
        output = input.view(-1, input.size(-1))  # Reshape to (N, in_size)
        batch = output.shape[0]
        if self.param == 'regular':
            output = output.expand((batch, self.in_size) + (()) )  #expand((batch, self.in_size)
        
        return output
    
    def deform_post_process(self, input, output):
        batch = output.shape[0]
        if self.bias is not None:
            output = output + self.bias
         
        return output.view(batch, self.out_size)

    def extra_repr(self):
        s = 'in_size={}, out_size={}, R_shapes={}, bias={}, param={}'.format(self.in_size, self.out_size, self.R_shapes, self.bias is not None, self.param)
        return s