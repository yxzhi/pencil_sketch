# -*- coding: utf-8 -*-
"""
@author: Menghao Zhao && Xiaozhi Yu

"""
import matplotlib.image as mpimg
import numpy as np
import scipy.signal as ss
from skimage.transform import resize
from skimage import color
from scipy import misc as ms
import scipy.sparse as ssparse
from scipy.sparse import linalg
import scipy.misc

def heaviside(x):
    x_toreturn = np.zeros(x.shape)
    for idx in range(len(x)):
        if x[idx] >= 0:
            x_toreturn[idx] = x[idx]
    return x_toreturn

def natural_histogram_matching(I,T):
    ho = np.zeros(256)
    po = np.zeros(256)
    I = (I*256).astype('uint8')
    for i in range(256):
        po[i] = np.sum(I == i)
    po = po / np.sum(po)
    ho[0] = po[0]
    for i in range(1, 256):
        ho[i] = ho[i - 1] + po[i]

    x = np.array([i for i in range(256)])

    p1 = np.exp(-(256-x)/9) * heaviside(256 - x)/9
    p2 = (heaviside(x - 105) - heaviside(x - 256)) / 151
    p3 = np.exp(-((x-90)**2)/242)/np.sqrt(22*np.pi)

    if T=='colour':
        p = 62*p1+30*p2+5*p3
    else:
        p = 76*p1+22*p2+2*p3
    prob = np.zeros(256)
    histo = np.zeros(256)
    for i in range(256):
        prob[i] = p[i]
    prob = prob/np.sum(prob)
    histo[0] = prob[0]
    for i in range(1, 256):
        histo[i] = histo[i-1]+prob[i]

    Iadjusted = np.zeros([len(I),len(I[0])])
    for y in range(len(I)):
        for x in range(len(I[0])):
            histogram_value = ho[I[y, x]]
            idx = np.argmin(abs(histo - histogram_value))
            Iadjusted[y, x] = idx
    return Iadjusted/np.max(Iadjusted)
        
def vertical_stitch(I, maxh):
    Istitch = I
    while len(Istitch) < maxh:
        quartersize = round(len(I) * 0.25)
        Iup = (I[len(I)-quartersize:,:]).astype('float32')
        Idown = (I[:quartersize,:]).astype('float32')
        aup = np.zeros([quartersize,len(Iup[0])])
        adown = np.zeros([quartersize,len(Iup[0])])
        for x in range(quartersize):
            aup[x,:] = Iup[x,:]*(1-(x + 1)/quartersize)
            adown[x,:] = Idown[x,:]*(x + 1)/quartersize
        Itemp = np.concatenate((np.concatenate((Istitch[0:len(Istitch)-quartersize,:], \
                aup+adown),axis = 0),Istitch[quartersize:,:]),axis = 0)
        Istitch = Itemp
    Istitch = Istitch[:maxh,:]
    return Istitch

def horizontal_stitch(I,maxw):
    Istitch = I
    while len(Istitch[0])<maxw:
        quartersize = round(len(I[0])*0.25)
        left = (I[:,len(I[0])-quartersize:]).astype('float32')
        right = (I[:,0:quartersize]).astype('float32')
        aleft = np.zeros([len(left),quartersize])
        aright = np.zeros([len(left),quartersize])
        for y in range(quartersize):
            aleft[:,y] = left[:,y]*(1-y/quartersize)
            aright[:,y] = right[:,y]*y/quartersize
        Itemp = np.concatenate((np.concatenate((Istitch[:,0:len(Istitch[0])-quartersize], \
            aleft+aright),axis = 1),Istitch[:,quartersize:]),axis=1);
        Istitch = Itemp
    Istitch = Istitch[:,:maxw]
    return Istitch

def rot90(mat):
    ret = np.zeros([len(mat[0]),len(mat)])
    for x in range(len(mat)):
        for y in range(len(mat[0])):
            ret[len(mat[0])-1-y,x] = mat[x,y]
    return ret

def pencil_draw(I, texture):
    line_len_divisor = 40#40
    line_thickness_divisor = 8
    lambdaa = 2#0.2
    texture_resize_ratio = 0.5

    #I = I.transpose((2,0,1))
    #if(len(I[0][0])==3):
    if np.max(I) > 2:
        I = I.astype('float32')/255

    if len(I.shape) == 3:
        J = color.rgb2gray(I)
        T = 'colour'
    else:
        J = I.astype('float32')
        T = 'black'
#---------------------------------------------------------------------------------------------#

    line_len_double = min(len(J), len(J[0])) / line_len_divisor

    if np.floor(line_len_double) % 2 == 1:
        line_len = int(np.floor(line_len_double))
    else:
        line_len = int(np.floor(line_len_double) + 1)
    half_line_len = (line_len + 1) / 2
    Ix = ss.convolve2d(J, np.array([[1, -1], [1, -1]]), mode = 'same')
    Iy = ss.convolve2d(J, np.array([[1, 1], [-1, -1]]), mode = 'same')
    Imag = np.sqrt(Ix * Ix + Iy * Iy)
    
    L = np.zeros([8, line_len,line_len])
    for n in range(8):
        if n < 4:
            for x in range(line_len):
                y  = int(half_line_len - round((x-half_line_len)*np.tan(np.pi*n/8)))
                if 0 <= y < line_len:
                    L[n, y, x] = 1
        else:
            L[n] = rot90(L[n - 4])
    
    valid_width = len(ss.convolve2d(L[0],\
                np.ones([round(line_len/line_thickness_divisor), \
                round(line_len/line_thickness_divisor)]),mode = 'valid'))
    Lthick = np.zeros([8, valid_width, valid_width])
    for n in range(8):
        Ln = ss.convolve2d(L[n],np.ones([round(line_len/line_thickness_divisor), \
                    round(line_len/line_thickness_divisor)]), mode='valid')
        Lthick[n] = Ln / np.max(Ln)
    
    G = np.zeros([8, len(J),len(J[0])])
    for n in range(8):
        G[n] = ss.convolve2d(Imag,L[n],mode = 'same')
    Gindex = G.argmax(0)
    C = np.zeros([8, len(J), len(J[0])])
    for n in range(8):
        C[n] = Imag * ((Gindex == n).astype(np.int))

    Spn = np.zeros([8, len(J), len(J[0])])

    for n in range(8):
        Spn[n] = ss.convolve2d(C[n], Lthick[n], mode='same')
    Sp = np.sum(Spn,0)
    Sp = (Sp-np.min(Sp))/(np.max(Sp)-np.min(Sp))
    S = 1 - Sp
#-----------------------------------------------------------------------------------------#
    Jadjusted = natural_histogram_matching(J, T)
    
    #texture = mpimg.imread(texture_file_name)
    if len(texture.shape) == 3:
        texture = color.rgb2gray(texture)
    #texture = texture.transpose((2, 0, 1))
    texture = texture[100:len(texture)-99,100:len(texture[0])-99]
    #texture = resize(texture,texture_resize_ratio / 1024.0 * np.min(len(J),len(J[0])))
    #texture = resize(texture,(np.round(texture_resize_ratio * min(len(J),len(J[0])) * len(texture) /1024),
    #                          np.round(texture_resize_ratio * min(len(J),len(J[0])) * len(texture[0]) / 1024))).astype(float)
    texture = scipy.misc.imresize(texture, (int(texture_resize_ratio * min(len(J),len(J[0])) * len(texture) /1024),
                                            int(texture_resize_ratio * min(len(J),len(J[0])) * len(texture[0]) / 1024)))
    texture = (texture.astype('float32')/255)

    Jtexture = vertical_stitch(horizontal_stitch(texture,len(J[0])),len(J))
    
    sizz = len(J) * len(J[0])
    nzmax = 2*(sizz-1)
    i = np.zeros(nzmax)
    j = np.zeros(nzmax)
    s = np.zeros(nzmax)
    for m in range(nzmax):
        i[m] = np.ceil((m + 1 + 0.1)/2)
        j[m] = np.ceil((m + 1 - 0.1)/2)
        s[m] = -2 * ((m + 1) % 2) + 1

    dx = ssparse.coo_matrix((s, (i - 1, j - 1)), (sizz, sizz))
    dx = dx.tocsr()
    nzmax = 2 * (sizz - len(J[0]))

    i = np.zeros(nzmax)
    j = np.zeros(nzmax)
    s = np.zeros(nzmax)
    for m in range(nzmax):
        if (m + 1) % 2 == 1:
            i[m] = np.ceil((m + 1.1)/2)
        else:
            i[m] = np.ceil((m + 0.1)/2) + len(J[0])
        j[m] = np.ceil((m + 1 - 0.1)/2)
        s[m] = -2 * ((m + 1) % 2) + 1

    dy = ssparse.coo_matrix((s, (i - 1, j - 1)), (sizz, sizz))
    dy = dy.tocsr()

    Jtexture1d = np.log(Jtexture.flatten())
    Jtsparse = ssparse.spdiags(Jtexture1d, 0, sizz, sizz)
    Jtsparse = Jtsparse.tocsr()

    Jadjusted1d = np.log(Jadjusted.flatten())
    beta1d = linalg.spsolve((Jtsparse.T.dot(Jtsparse) + \
              lambdaa * (dx.T.dot(dx) + dy.T.dot(dy))), Jtsparse.dot(Jadjusted1d))
    beta = np.reshape(beta1d, (len(J), len(J[0])))

    T = Jtexture ** beta
    T = (T - np.min(T)) / (np.max(T) - np.min(T))
    return S * T

def pencil_draw_color(I, texture):
    if len(I.shape) == 2:
        return pencil_draw(I, texture)
    else:
        I_yuv = color.rgb2yuv(I)
        Y_gray = pencil_draw(I_yuv[:,:,0], texture)

        I_yuv[:,:,0] = Y_gray
        I_after= color.yuv2rgb(I_yuv)
        I_after = np.maximum(I_after, 0)
        I_after = np.minimum(I_after, 1)
        #I_ruv[:, :, 0] = 0.5
        return I_after
