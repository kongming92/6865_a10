#Light field Assignment
#By Abe Davis

import numpy as np
from scipy import ndimage

#   ###  *NOTES!*  ###
# Our light fields are to be indexed
# LF[v,u,y,x,color]
#
# Our focal stacks are to be indexed
# FS[image, y, x, color]

def apertureView(LF):
    '''Takes a light field, returns 'out,' an image with nx*ny sub-pictures representing the value of each pixel in each of the nu*nv views.'''
    nv = LF.shape[0]
    nu = LF.shape[1]
    ny = LF.shape[2]
    nx = LF.shape[3]
    out = np.zeros([nv*ny, nu*nx, 3])
    for v in xrange(nv):
        for u in xrange(nu):
            out[v::nv,u::nu] = LF[v, u]
    return out

def epiSlice(LF, y):
    '''Takes a light field. Returns the epipolar slice with constant v=(nv/2) and constant y (input argument).'''
    return LF[LF.shape[0]/2, :, y, :, :]

def shiftFloat(im, dy, dx):
    '''Returns a copy of im shifted by the floating point values dy in y and dx in x.
    We want this to be fast so use either scipy.ndimage.map_coordinates or scipy.ndimage.interpolation.affine_transform.'''
    return ndimage.interpolation.shift(im, [dy, dx, 0], mode='nearest')

def refocusLF(LF, maxParallax=0.0, aperture=17):
    '''Takes a light field as input and outputs a focused image by summing over u and v with the correct shifts applied to each image.
    Use aperture*aperture views, centered at the center of views in LF.
    A view at the center should not be shifted.
    Views at opposite ends of the aperture should have shifts that differ by maxParallax. See handout for more details.'''
    Nv = LF.shape[0]
    Nu = LF.shape[1]
    centerv = Nv/2 #v coordinate of center view
    centeru = Nu/2 #u coordinate of center view
    width = int(aperture/2)
    out = np.zeros([LF.shape[2], LF.shape[3], 3])
    scale = float(2*maxParallax) / (aperture-1)
    for v in xrange(centerv - width, centerv + width + 1):
        for u in xrange(centeru - width, centeru + width + 1):
            dy = (v - centerv) * scale
            dx = -(u - centeru) * scale
            out += shiftFloat(LF[v, u], dy, dx)
    return out / (Nv*Nu)

def rackFocus(LF, aperture=8, nIms = 15, minmaxPara=-7.0, maxmaxPara=2.0):
    '''Takes a light field, returns a focal stack. See handout for more details '''
    out = np.zeros([nIms, LF.shape[2], LF.shape[3], LF.shape[4]])
    increment = (maxmaxPara - minmaxPara) / (nIms - 1)
    for i in xrange(nIms):
        parallax = minmaxPara + i * increment
        out[i] = refocusLF(LF, parallax, aperture)
    return out

def sharpnessMap(im, exponent=1.0, sigma=1.0):
    '''Computes the sharpness map of one image. This will be used when we compute all-focus images. See handout.'''
    bw = np.dot(im, np.array([0.3, 0.6, 0.1]))
    highpass = bw - ndimage.filters.gaussian_filter(bw, sigma)
    high_freq_energy = ndimage.filters.gaussian_filter(highpass * highpass, 4 * sigma)
    high_freq_energy_sharp = high_freq_energy ** exponent
    out = np.zeros(im.shape)
    for i in xrange(3):
        out[:,:,i] = high_freq_energy_sharp[:,:]
    return out

def sharpnessStack(FS, exponent=1.0, sigma=1.0):
    '''This should take a focal stack and return a stack of sharpness maps.
    We provide this function for you.'''
    SS = np.zeros_like(FS)
    for i in xrange(FS.shape[0]):
        SS[i]=sharpnessMap(FS[i], exponent, sigma)
    return SS

def fullFocusLinear(stack, exponent=1.0, sigma=1.0):
    '''takes a numpy array stack[image, y, x, color] and returns an all-focus image and a depth map. See handout.'''
    N = stack.shape[0]
    out = np.zeros_like(stack[0])
    zmap = np.zeros_like(stack[0])
    sharpness = sharpnessStack(stack, exponent, sigma)
    weight = np.zeros_like(stack[0])
    for i in xrange(N):
        out += stack[i] * sharpness[i]
        zmap += float(i) / (N-1) * sharpness[i]
        weight += sharpness[i]
    return out / weight, (zmap / weight)**2
