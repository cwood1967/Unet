import numpy as np
from scipy.ndimage import rotate

def check_extent(n, f, e):
    if f > n:
        d = f - n
        f = n
        e = e - d
    return (e, f)

def makes_patches(image, patch_shape=(8, 256, 256)):
    nz, nc, ny, nx = image.shape
    ez = 0
    ex = 0
    ey = 0

    dz, dy, dx = patch_shape
    patches = list()
    while ez < nz:
        ex = 0
        ey = 0
        fz = ez + dz
        ez, fz = check_extent(nz, fz, ez)
        print(ez)
        while ey < ny:
            ex = 0
            fy = ey + dy
            ey, fy = check_extent(ny, fy, ey)
            #print('y', ey, fy)
            while ex < nx:
                fx = ex + dx
                ex, fx = check_extent(nx, fx, ex)
                #print('x', ex, fx, ey, fy, ez, fz) 
                patch = image[ez:fz, :, ey:fy, ex:fx]
                #print(ex, ey, ez, patch.max(axis=(0,2,3)))
                patches.append(patch)
                ex += dx
            ey += dy

        ez += dz
        #print(ez)
        a = np.stack(patches)
    return a
    

def augment_rotate(stack, step, axes=(-1, -2)):
    
    angle = step
    res = stack.copy()
    while angle < 360:
        print(angle)
        r = rotate(stack, step, axes=axes, reshape=False)
        res = np.concatenate([res, r], axis=0)
        print(res.shape)
        angle += step

    return res #np.concatenate([stack, r], axis = 0)

        
def reconstruct(patches, w, nx, ny):
    
    image = np.zeros((ny, nx, patches.shape[-1]), dtype=patches.dtype)
    xmax = 0
    ymax = 0
    xok = True
    yok = True
    patch_index = 0
    while xok:
        ymax = 0
        yok = True
        xs = xmax
        xmax += w
        if xmax >= nx:
            xmax = nx
            xs = nx - w
            xok = False
        while yok:
            ys = ymax
            ymax += w
            if ymax > ny:
                ymax = ny
                ys = ny - w
                yok = False
            
            crop = 32
            image[ys:ymax, xs:xmax, :] = patches[patch_index]
            image[ys:ys + crop, :] = 0
            image[:, xs:xs + crop, :] = 0
            image[ymax - crop:ymax, :] = 0
            image[:, xmax-crop:xmax] = 0
            patch_index += 1
            #print(patch_index, ys, ymax, xs, xmax, yok, xok)
    return image

def patches_to_image(patches, w, nx, ny):
    r1 = patches_to_image(dp, 256, data.shape[1] , data.shape[0])
    r2 = patches_to_image(dp2, 256, data.shape[1] - 0, data.shape[0] - 128)
    r3 = patches_to_image(dp3, 256, data.shape[1] - 128, data.shape[0] - 0)
    r4 = patches_to_image(dp4, 256, data.shape[1] - 128, data.shape[0] - 128)

    rr2 = np.zeros_like(r1)
    rr2[128:, :, :] = r2

    rr3 = np.zeros_like(r1)
    rr3[:, 128:, :] = r3

    rr4 = np.zeros_like(r1)
    rr4[128:, 128:, :] = r4

    rstack = np.stack((r1, rr2, rr3, rr4), axis=0)

    r = rstack.max(axis=0)

