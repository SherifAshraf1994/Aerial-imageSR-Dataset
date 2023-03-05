import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
import os

def simulate_satellite_degradation(image, noise_sigma=5, blur_kernel_size=5,Camera_Filter_size=7,Camera_sigma = 0.6, compression_quality=80,HrFlag = False,scale_factor=2):
    if HrFlag == False:
        # Add Gaussian noise to simulate low light conditions
        noise = np.random.normal(0, noise_sigma, image.shape)
        degraded_image = image + noise
        # Apply motion blur to simulate the movement of the satellite
        blur_kernel = np.ones((blur_kernel_size, blur_kernel_size)) / blur_kernel_size ** 2
        degraded_image = cv2.filter2D(degraded_image, -1, blur_kernel)
        # Apply gaussian filter to simulate camera lens
        degraded_image = gaussian_filter(degraded_image, sigma=Camera_sigma, order=0, mode='reflect', cval=0.0, truncate=Camera_Filter_size)
        
        # Apply subsampling to simulate low resolution
        degraded_image = degraded_image[::scale_factor,::scale_factor,:]
        
        # Compress the image to simulate data transmission
        _, compressed_image = cv2.imencode(".jpg", degraded_image, [int(cv2.IMWRITE_JPEG_QUALITY), compression_quality])
        degraded_image = cv2.imdecode(compressed_image, -1)
    if HrFlag == True:
        hr_filterd = gaussian_filter(image, sigma=Camera_sigma, order=0, mode='reflect', cval=0.0, truncate=Camera_Filter_size)
        degraded_image = hr_filterd[::scale_factor,::scale_factor,:]
    return degraded_image





# filter size 9 HR, 17 LR2, 33 LR4, 65 LR8
#sigma for filter 0.5 HR, 0.6 LR2, 0.7 LR4, 0.8 LR8

ContinousScenePath = "Raw\Dataset\Path"
datasetList = os.listdir(ContinousScenePath)
for imageName in datasetList:
    ContinousScene = cv2.imread(os.path.join(ContinousScenePath,imageName))
    HrImage = simulate_satellite_degradation(ContinousScene,Camera_Filter_size=9,Camera_sigma=0.5,HrFlag=True,scale_factor=2)
    cv2.imwrite("Your\DatasetPath\HR\\"+imageName,HrImage)
    Lr2Image = simulate_satellite_degradation(ContinousScene,Camera_Filter_size=17,Camera_sigma=0.6,HrFlag=False,scale_factor=4)
    cv2.imwrite("Your\DatasetPath\LR2\\"+imageName,Lr2Image)
    Lr4Image = simulate_satellite_degradation(ContinousScene,Camera_Filter_size=33,Camera_sigma=0.7,HrFlag=False,scale_factor=8)
    cv2.imwrite("Your\DatasetPath\LR4\\"+imageName,Lr4Image)
    Lr8Image = simulate_satellite_degradation(ContinousScene,Camera_Filter_size=65,Camera_sigma=0.8,HrFlag=False,scale_factor=16)
    cv2.imwrite("Your\DatasetPath\LR8\\"+imageName,Lr8Image)
    print(imageName)
print("done")
