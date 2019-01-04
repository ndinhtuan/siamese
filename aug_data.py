from imgaug import augmenters as iaa
import numpy as np 
import cv2 

class AugData(object):

    def __init__(self):
        pass 

    #images is list of image (w,h,channels) of (N, w, h, c)
    # output is similar so can train 
    def augmentData(self, images):
        
        sometimes = lambda aug: iaa.Sometimes(0.5, aug)

        augumenters = iaa.Sequential([
            iaa.SomeOf(n=(0, 5), children=[
                # iaa.Superpixe-ls(p_replace=(0.1, 1.0), n_segments=(10, 120)),
                iaa.Invert(0.2, 0.5),
                iaa.Add(value=(-40, 40), per_channel=0.5),
                iaa.Multiply(mul=(0.5, 1.5), per_channel=0.5),
                iaa.OneOf([
                    iaa.GaussianBlur(sigma=(0.0, 0.5)),
                    iaa.AverageBlur(k=(3, 15)),
                    iaa.MedianBlur(k=(3, 15))
                ]),
                iaa.AdditiveGaussianNoise(scale=0.05*255, per_channel=0.5),
                iaa.Dropout(p=(0, 0.2), per_channel=0.5),
                iaa.Grayscale(alpha=(0.0, 1.0)),
                iaa.ContrastNormalization((0.5, 1.5), per_channel=0.5),
                iaa.Sharpen(alpha=(0.0, 1.0), lightness=(0.75, 2.0)),
                iaa.Affine(scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)} , rotate=(-3, 3), shear=(-5, 5))
            ], random_order=True),
            sometimes(iaa.CropAndPad(percent=(-0.05, 0.1), pad_mode=["constant", "edge"], pad_cval=(0, 128)))
        ], random_order=False)
        augumenters = augumenters.to_deterministic() #
        newDatas = augumenters.augment_images(images)
        return newDatas # -> dua vao train binh thuong

def soft_aug(imgs):
    aug = iaa.Sequential(
            [iaa.SomeOf(n=(1), children=[
                iaa.MedianBlur(k=(3,5)),
                iaa.AdditiveGaussianNoise(scale=0.05*255),
                iaa.GaussianBlur(sigma=(0.4, 0.7)),
                iaa.Add(value=(-40, 40), per_channel=0.5),
                iaa.Affine(scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)} , rotate=(-3, 3), shear=(-5, 5))])
            ])
    
    return aug.augment_image(imgs)

if __name__ == "__main__":

    img = cv2.imread("orl_faces/s1/1.pgm")

    while True :
        tmp = soft_aug(img)
        horizontal_concat = np.hstack((img, tmp))
        cv2.namedWindow("demo", cv2.WINDOW_NORMAL)
        cv2.imshow("demo", horizontal_concat)
        cv2.waitKey(0)
