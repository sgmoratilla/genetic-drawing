import cv2
import numpy as np
import time
import random

class Stroke:
    def __init__(self, color, posY, posX, size, rotation, brushNumber):
        self.color = color
        self.posY = posY
        self.posX = posX
        self.size = size
        self.rotation = rotation
        self.brushNumber = brushNumber
    

class ImageDrawer:
    def __init__(self, image_shape=None, canvas=None, sampling_mask=None, n_brushes=4):
        if image_shape is None and canvas is None:
            raise ValueError("either image_shape or canvas must be specified")

        self.image_shape = image_shape
        self.canvas = canvas

        if self.image_shape is None:
            self.image_shape = self.canvas.shape

        if self.canvas is None:
            self.canvas = np.zeros((self.image_shape[0], self.image_shape[1]), np.uint8)

        self.sampling_mask = sampling_mask
        self.n_brushes = n_brushes

        self.brushes = self.preload_brushes('brushes/watercolor/', self.n_brushes)
        
    def preload_brushes(self, path, n_brushes):
        imgs = []
        for i in range(n_brushes):
            imgs.append(cv2.imread(path + str(i) +'.jpg'))
        return imgs

    def draw(self, strokes, padding):
        # Don't overwrite the original canvas
        canvas = np.copy(self.canvas)

        # apply padding
        p = padding
        canvas = cv2.copyMakeBorder(canvas, p,p,p,p,cv2.BORDER_CONSTANT,value=[0,0,0])
        # draw every stroke
        for i in range(len(strokes)):
            canvas = self.__drawStroke(strokes[i], padding, canvas)
        # remove padding
        y = canvas.shape[0]
        x = canvas.shape[1]

        return canvas[p:(y-p), p:(x-p)]

    def __drawStroke(self, stroke, padding, canvas):
        # get DNA data
        color = stroke.color
        posX = int(stroke.posX) + padding # add padding since indices have shifted
        posY = int(stroke.posY) + padding
        size = stroke.size
        rotation = stroke.rotation
        brushNumber = int(stroke.brushNumber)

        # load brush alpha
        brushImg = self.brushes[brushNumber]
        # resize the brush
        brushImg = cv2.resize(brushImg,None,fx=size, fy=size, interpolation = cv2.INTER_CUBIC)
        # rotate
        brushImg = rotate_image(brushImg, rotation)
        # brush img data
        brushImg = cv2.cvtColor(brushImg,cv2.COLOR_BGR2GRAY)
        rows, cols = brushImg.shape
        
        # create a colored canvas
        myClr = np.copy(brushImg)
        myClr[:, :] = color

        # find ROI
        inImg_rows, inImg_cols = canvas.shape
        y_min = int(posY - rows/2)
        y_max = int(posY + (rows - rows/2))
        x_min = int(posX - cols/2)
        x_max = int(posX + (cols - cols/2))
        
        # Convert uint8 to float
        foreground = myClr[0:rows, 0:cols].astype(float)
        background = canvas[y_min:y_max,x_min:x_max].astype(float) #get ROI
        # Normalize the alpha mask to keep intensity between 0 and 1
        alpha = brushImg.astype(float)/255.0
        
        try:
            # Multiply the foreground with the alpha matte
            foreground = cv2.multiply(alpha, foreground)
            
            # Multiply the background with ( 1 - alpha )
            background = cv2.multiply(np.clip((1.0 - alpha), 0.0, 1.0), background)
            # Add the masked foreground and background.
            outImage = (np.clip(cv2.add(foreground, background), 0.0, 255.0)).astype(np.uint8)
            
            canvas[y_min:y_max, x_min:x_max] = outImage
        except:
            print('------ \n', 'in image ',canvas.shape)
            print('pivot: ', posY, posX)
            print('brush shape: ', brushImg.shape)
            print('fg: ', foreground.shape)
            print('bg: ', background.shape)
            print('alpha: ', alpha.shape)
        
        return canvas


def sample_from_img(img):
    # possible positions to sample
    pos = np.indices(dimensions=img.shape)
    pos = pos.reshape(2, pos.shape[1]*pos.shape[2])
    img_flat = np.clip(img.flatten() / img.flatten().sum(), 0.0, 1.0)
    return pos[:, np.random.choice(np.arange(pos.shape[1]), 1, p=img_flat)]
        
def rotate_image(img, angle):
    rows,cols, channels = img.shape
    M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
    dst = cv2.warpAffine(img,M,(cols,rows))
    return dst
        
'''
we'd like to "guide" the brushtrokes along the image gradient direction, if such direction has large magnitude
in places of low magnitude, we allow for more deviation from the direction. 
this function precalculates angles and their magnitudes for later use inside DNA class
'''
def image_gradient(img):
    #convert to 0 to 1 float representation
    img = np.float32(img) / 255.0 
    # Calculate gradient 
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1)
    # Python Calculate gradient magnitude and direction ( in degrees ) 
    mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    #normalize magnitudes
    mag /= np.max(mag)
    #lower contrast
    mag = np.power(mag, 0.3)
    return mag, angle