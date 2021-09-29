# AutoChain Generation
## Core codes
```python
'''
This is a demo to show how to generate chains automatically, it contains:
1) A function to generate the required chains
2) A function to fill the matrix with zeros to make the height/width factorizable.

We conduct experiments under 5 different settings:
1) factorizable #channels + fast shrinking + bulging chains
2) factorizable #channels + slow shrinking + bulging chains
3) factorizable #channels + fast shrinking + monotonic chains
4) factorizable #channels + slow shrinking + monotonic chains
5) Padding a matrix to make the #channels factorizable

Considering the network design in the reality, this demo assumes that the spatial
size of the kernel is 3x3, and the number of input and output channels for each
layer will not change steeply.

Anonymous authors of DeBut, 10/08/2021.
'''
import numpy as np
import math

# basic chain
def AutoChain(matrix, type='monotonic', shrinking_speed=5):
    '''
    Input:
        - matrix: a matrix, which is of size [c_out, k*c_in].
        - type: a string, which indicates the type of chain.
        - shrinking_speed: an integer, which decides the shrinking speed of the chain.
        Larger the number, slower the shrinking speed. we suggest to set shrinking_speed
        in the interval [4,8]. Please set this parameter properly according to the matrix
        size.

    Return:
        - sup: a list, which is the superscripts of the factors from right to left.
        - sub: a list, which is the subscripts of the factors from right to left.

    The parameters r, s can be modified as well.
    '''
    h, w = matrix.shape
    w_channel = w / (3*3)
    sup = []
    sub = []
    if type == 'monotonic':
        if h == w_channel:
            sup.append([int(math.pow(2, 3+ np.ceil(np.log2(w_channel)))), w])
            sub.append([int(math.pow(2, 3)), 3*3, 1])
            for i in range(shrinking_speed-5):
                sup.append([int(math.pow(2, 3+ np.ceil(np.log2(w_channel)))), int(math.pow(2, 3+ np.ceil(np.log2(w_channel))))])
                rt = sub[-1][0] * sub[-1][2]
                r = 2
                s = 2
                sub.append([r, s, rt])
            for j in range(2):
                sup.append([int(math.pow(2, 3+ np.ceil(np.log2(w_channel))-j-1)), int(math.pow(2, 3+np.ceil(np.log2(w_channel))-j))])
                rt = sub[-1][0] * sub[-1][2]
                r = 2
                s = 4
                sub.append([r, s, rt])
            sup.append([int(h), int(h*2)])
            rt = sub[-1][0] * sub[-1][2]
            r = h / rt
            s = h*2 / rt
            sub.append([int(r), int(s), int(rt)])
        elif h == w_channel / 2:
            sup.append([int(math.pow(2, 3+ np.ceil(np.log2(w_channel)))), w])
            sub.append([int(math.pow(2, 3)), 3*3, 1])
            for i in range(shrinking_speed-5):
                sup.append([int(math.pow(2, 3+ np.ceil(np.log2(w_channel)))), int(math.pow(2, 3+ np.ceil(np.log2(w_channel))))])
                rt = rt = sub[-1][0] * sub[-1][2]
                r = 2
                s = 2
                sub.append([r, s, rt])
            for j in range(3):
                sup.append([int(math.pow(2, 3+ np.ceil(np.log2(w_channel))-j-1)), int(math.pow(2, 3+np.ceil(np.log2(w_channel))-j))])
                rt = sub[-1][0] * sub[-1][2]
                r = 2
                s = 4
                sub.append([r, s, rt])
            sup.append([int(h), int(h*2)])
            rt = sub[-1][0] * sub[-1][2]
            r = h / rt
            s = h*2 / rt
            sub.append([int(r), int(s), int(rt)])
        elif h == w_channel * 2:
            sup.append([int(math.pow(2, 3+ np.ceil(np.log2(w_channel)))), w])
            sub.append([int(math.pow(2, 3)), 3*3, 1])
            for i in range(shrinking_speed-3):
                sup.append([int(math.pow(2, 3+ np.ceil(np.log2(w_channel)))), int(math.pow(2, 3+ np.ceil(np.log2(w_channel))))])
                rt = rt = sub[-1][0] * sub[-1][2]
                r = 2
                s = 2
                sub.append([r, s, rt])
            for j in range(1):
                sup.append([int(math.pow(2, 3+ np.ceil(np.log2(w_channel))-j-1)), int(math.pow(2, 3+np.ceil(np.log2(w_channel))-j))])
                rt = sub[-1][0] * sub[-1][2]
                r = 2
                s = 4
                sub.append([r, s, rt])
            sup.append([int(h), int(h*2)])
            rt = sub[-1][0] * sub[-1][2]
            r = h / rt
            s = h*2 / rt
            sub.append([int(r), int(s), int(rt)])
    elif type == 'bulging':
        bulging_rate = 4/3
        sup.append([int(math.pow(2, 3+ np.ceil(np.log2(w_channel))) * bulging_rate), w])
        rt = 1
        s = 6
        r = s * bulging_rate
        sub.append([int(r), int(s), int(rt)])
        sup.append([int(math.pow(2, 3+ np.ceil(np.log2(w_channel)))), int(math.pow(2, 3+ np.ceil(np.log2(w_channel))) * bulging_rate)])
        s = 6
        r = s * bulging_rate / (2*2)
        rt = sub[-1][0] * sub[-1][2]
        sub.append([int(r), int(s), int(rt)])
        if h == w_channel:
            for i in range(shrinking_speed-5-1):
                sup.append([int(math.pow(2, 3+ np.ceil(np.log2(w_channel)))), int(math.pow(2, 3+ np.ceil(np.log2(w_channel))))])
                rt = sub[-1][0] * sub[-1][2]
                r = 2
                s = 2
                sub.append([r, s, rt])
            for j in range(2):
                sup.append([int(math.pow(2, 3+ np.ceil(np.log2(w_channel))-j-1)), int(math.pow(2, 3+np.ceil(np.log2(w_channel))-j))])
                rt = sub[-1][0] * sub[-1][2]
                r = 2
                s = 4
                sub.append([r, s, rt])
            sup.append([int(h), int(h*2)])
            rt = sub[-1][0] * sub[-1][2]
            r = h / rt
            s = h*2 / rt
            sub.append([int(r), int(s), int(rt)])
        elif h == w_channel / 2:
            for i in range(shrinking_speed-5-1):
                sup.append([int(math.pow(2, 3+ np.ceil(np.log2(w_channel)))), int(math.pow(2, 3+ np.ceil(np.log2(w_channel))))])
                rt = rt = sub[-1][0] * sub[-1][2]
                r = 2
                s = 2
                sub.append([r, s, rt])
            for j in range(3):
                sup.append([int(math.pow(2, 3+ np.ceil(np.log2(w_channel))-j-1)), int(math.pow(2, 3+np.ceil(np.log2(w_channel))-j))])
                rt = sub[-1][0] * sub[-1][2]
                r = 2
                s = 4
                sub.append([r, s, rt])
            sup.append([int(h), int(h*2)])
            rt = sub[-1][0] * sub[-1][2]
            r = h / rt
            s = h*2 / rt
            sub.append([int(r), int(s), int(rt)])
        elif h == w_channel * 2:
            for i in range(shrinking_speed-3-1):
                sup.append([int(math.pow(2, 3+ np.ceil(np.log2(w_channel)))), int(math.pow(2, 3+ np.ceil(np.log2(w_channel))))])
                rt = rt = sub[-1][0] * sub[-1][2]
                r = 2
                s = 2
                sub.append([r, s, rt])
            for j in range(1):
                sup.append([int(math.pow(2, 3+ np.ceil(np.log2(w_channel))-j-1)), int(math.pow(2, 3+np.ceil(np.log2(w_channel))-j))])
                rt = sub[-1][0] * sub[-1][2]
                r = 2
                s = 4
                sub.append([r, s, rt])
            sup.append([int(h), int(h*2)])
            rt = sub[-1][0] * sub[-1][2]
            r = h / rt
            s = h*2 / rt
            sub.append([int(r), int(s), int(rt)])
    else:
        assert 'Only monotonic or bulging chains!'
    print('Chain Type: {}.'.format(type))
    print('Shrinking Speed: {}.'.format(shrinking_speed))
    print('Number of factors: {}.'.format(sub))
    print('Superscript: {}.'.format(sup))
    print('Subscript: {}.'.format(sub))
    return sup, sub

# padding function
def PaddingMatrix(matrix1):
    '''
    Input:
        - matrix1: a matrix, whose height/width is not factorizable
    Output:
        - matrix2: a matrix, whose height/width is factorizable.
        compared with matrix1, the additional elements are zeros.
    '''
    h, w = matrix1.shape
    w_channel = w / (3*3)
    h_hat = math.pow(2, np.ceil((np.log2(h))))
    w_hat = math.pow(2, np.ceil((np.log2(w_channel)))) * (3*3)
    matrix2 = np.zeros([int(h_hat), int(w_hat)])
    matrix2[:h, :w] = matrix1
    print('The size of the flattened matrix now: {}.'.format(matrix2.shape))
    return matrix2
```
## Demos

### demo 1: factorizable #channels + fast shrinking + bulging chains
```python
matrix = np.random.rand(256, 2304)
sup, sub = AutoChain(matrix, type='bulging', shrinking_speed=5)
```
The output:
```
Chain Type: bulging.
Shrinking Speed: 5.
Number of factors: [[8, 6, 1], [2, 6, 8], [2, 4, 16], [2, 4, 32], [4, 8, 64]].
Superscript: [[2730, 2304], [2048, 2730], [1024, 2048], [512, 1024], [256, 512]].
Subscript: [[8, 6, 1], [2, 6, 8], [2, 4, 16], [2, 4, 32], [4, 8, 64]].
```

### demo 2: demo 2: factorizable #channels + slow shrinking + bulging chanis
```python
matrix = np.random.rand(256, 2304)
sup, sub = AutoChain(matrix, type='bulging', shrinking_speed=8)
```
The output:
```
Chain Type: bulging.
Shrinking Speed: 8.
Number of factors: [[8, 6, 1], [2, 6, 8], [2, 2, 16], [2, 2, 32], [2, 4, 64], [2, 4, 128], [1, 2, 256]].
Superscript: [[2730, 2304], [2048, 2730], [2048, 2048], [2048, 2048], [1024, 2048], [512, 1024], [256, 512]].
Subscript: [[8, 6, 1], [2, 6, 8], [2, 2, 16], [2, 2, 32], [2, 4, 64], [2, 4, 128], [1, 2, 256]].
```

### demo 3: factorizable #channels + fast shrinking + monotonic chains
```python
matrix = np.random.rand(256, 2304)
sup, sub = AutoChain(matrix, type='monotonic', shrinking_speed=4)
```
The output:
```
Chain Type: monotonic.
Shrinking Speed: 4.
Number of factors: [[8, 9, 1], [2, 4, 8], [2, 4, 16], [8, 16, 32]].
Superscript: [[2048, 2304], [1024, 2048], [512, 1024], [256, 512]].
Subscript: [[8, 9, 1], [2, 4, 8], [2, 4, 16], [8, 16, 32]].
```

### demo 4: factorizable #channels + slow shrinking + monotonic chains
```python
matrix = np.random.rand(256, 2304)
sup, sub = AutoChain(matrix, type='monotonic', shrinking_speed=8)
```
The output:
```
Chain Type: monotonic.
Shrinking Speed: 8.
Number of factors: [[8, 9, 1], [2, 2, 8], [2, 2, 16], [2, 2, 32], [2, 4, 64], [2, 4, 128], [1, 2, 256]].
Superscript: [[2048, 2304], [2048, 2048], [2048, 2048], [2048, 2048], [1024, 2048], [512, 1024], [256, 512]].
Subscript: [[8, 9, 1], [2, 2, 8], [2, 2, 16], [2, 2, 32], [2, 4, 64], [2, 4, 128], [1, 2, 256]].
```

### demo 5: Padding a matrix to make the #channels factorizable
```python
matrix = np.random.rand(127,2231)
matrix2 = PaddingMatrix(matrix)
```
The output:
```
The size of the flattened matrix now: (128, 2304).
```