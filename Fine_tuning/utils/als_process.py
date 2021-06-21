import torch
import numpy as np

def process_als_weight(DeBut_factors, R_shapes):
    '''Process from left to right (input to output)'''
    v = []
    for i, m_i in enumerate(DeBut_factors[::-1]):
        R_i = R_shapes[i]
        output_size, input_size, row, col, diag = R_i[:]
        m_i = m_i.numpy()
        ele_i = m_i[m_i.nonzero()]
        ele_i = torch.from_numpy(ele_i)
        v_i = ele_i.view(input_size//(col*diag),row, diag, col).permute(0,2,1,3)#(input_size // (col * diag),row,diag,col).permute(0,2,1,3)
        v_i = v_i.reshape(-1)
    
        v.extend(v_i)
    return v