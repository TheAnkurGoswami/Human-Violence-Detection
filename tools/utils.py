import numpy as np
from functools import reduce

from PIL import Image


# find connection in the specified sequence, center 29 is in the position 15
limbSeq=[[2,3],[2,6],[3,4],[4,5],[6,7],[7,8],[2,9],[9,10],
         [10,11],[2,12],[12,13],[13,14],[2,1],[1,15],[15,17],
         [1,16],[16,18],[3,17],[6,18]]

# the middle joints heatmap correpondence
hmapIdx=[[31,32],[39,40],[33,34],[35,36],[41,42],[43,44],[19,20],[21,22],
         [23,24],[25,26],[27,28],[29,30],[47,48],[49,50],[53,54],[51,52],
         [55,56],[37,38],[45,46]]

# visualize
colors=[[255,0,0],[255,85,0],[255,170,0],[255,255,0],[170,255,0],[85,255,0],
        [0,255,0],
        [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255],
        [85, 0, 255],
        [170,0,255],[255, 0, 255], [255, 0, 170], [255, 0, 85]]


def pad_right_down_corner(img,stride,pad_value):   # 8   128
    h=img.shape[0]
    w=img.shape[1]

    pad=4*[None]
    pad[0]=0  # up
    pad[1]=0  # left
    pad[2]=0 if (h%stride==0) else stride-(h%stride)  # down
    pad[3]=0 if (w%stride==0) else stride-(w%stride)  # right

    img_padded=img
    pad_up=np.tile(img_padded[0:1,:,:]*0+pad_value,(pad[0],1,1))
    img_padded=np.concatenate((pad_up,img_padded),axis=0)
    pad_left=np.tile(img_padded[:,0:1,:]*0+pad_value,(1,pad[1],1))
    img_padded=np.concatenate((pad_left,img_padded),axis=1)
    pad_down=np.tile(img_padded[-2:-1,:,:]*0+pad_value,(pad[2],1,1))
    img_padded=np.concatenate((img_padded,pad_down),axis=0)
    pad_right=np.tile(img_padded[:,-2:-1,:]*0+pad_value,(1,pad[3],1))
    img_padded=np.concatenate((img_padded,pad_right),axis=1)

    return img_padded,pad

def compose(*funcs):
    #Compose arbitrarily many functions, evaluated left to right.
    # return lambda x: reduce(lambda v, f: f(v), funcs, x)
    if funcs:
        return reduce(lambda f,g: lambda *a,**kw: g(f(*a,**kw)),funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')

def letterbox_image(image,size):
    #Resize image with unchanged aspect ratio using padding.
    image_w,image_h=image.size
    w,h=size
    new_w=int(image_w*min(w*1.0/image_w,h*1.0/image_h))
    new_h=int(image_h*min(w*1.0/image_w,h*1.0/image_h))
    resized_image=image.resize((new_w,new_h),Image.BICUBIC)
    #Padding the image
    boxed_image=Image.new('RGB',size,(128,128,128))     #New image with a size & color(128,128,128)
    boxed_image.paste(resized_image,((w-new_w)//2,(h-new_h)//2))       #Pasting resized_image over newly created image at given coordinates.
    return boxed_image
