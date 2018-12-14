# Image-Segmentation-binarization-on-H-DIBCO-2016

Image Segmentation on H-DIBCO 2016 Dataset
Here image binarization is done on images with low quality,varying intyensity images
1)extract the dibco dataset(including ground truth images)
2)run finimg.py
3)After this running,an fscore of 86.5 is achieved.

Methods implemented are
1)Stauration removal
2)Noise removal using k-noise means method
3)image sharpning using kernel filter
4)Again Noise removal using k-noise means method
5)Normal image binarization
6)Blurring using a 3*3 kernel
7)Now at last Adaptive thresholding is used.
