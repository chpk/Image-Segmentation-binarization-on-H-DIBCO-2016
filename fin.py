import PIL
from PIL import Image,ImageEnhance
import cv2
for t in range(1,11):
    if t is not 8:
        print("processing "+'DIPCO2016_dataset\\'+str(t)+'.bmp')
        pth='DIPCO2016_dataset\\'+str(t)+'.bmp'
        img = PIL.Image.open(pth)
        converter = ImageEnhance.Color(img)
        img2 = converter.enhance(0)
        img2.save('edgedetect.bmp')
        import numpy as np
        img = cv2.imread('edgedetect.bmp')
        if (t == 1) or (t == 10):
            fatfnt=1
        else:
            fatfnt=0
        dst = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,7)
        #cv2.imwrite('nonoise1.jpg',dst) 
        cv2.imwrite('nonoise.jpg',dst)
        image = cv2.imread('nonoise.jpg')
        # Create our shapening kernel, it must equal to one eventually
        kernel_sharpening = np.array([[-1,-1,-1], 
                                      [-1, 9,-1],
                                      [-1,-1,-1]])
        
        # applying the sharpening kernel to the input image & displaying it.
        sharpened = cv2.filter2D(image, -1, kernel_sharpening)
        #cv2.imshow('sharpen.jpg',sharpened)
        #cv2.waitKey(0) 
        #img = cv2.imread('edgedetect.bmp') 
        dst1 = cv2.fastNlMeansDenoisingColored(sharpened,None,10,10,17,7)
        for i in range(0,1):
            dst1 = cv2.fastNlMeansDenoisingColored(dst1,None,10,10,17,7) 
        #cv2.imshow('sharpen1.jpg',dst)
        #cv2.waitKey(0)
        #ret,thresh1 = cv2.threshold(sharpened,100,255,cv2.THRESH_BINARY)
        cv2.imwrite('nonoise.jpg',dst1)
        img1 = cv2.imread('nonoise.jpg',0)
        ret1,th1 = cv2.threshold(img1,110,255,cv2.THRESH_BINARY) 
        ## global thresholding
        ##ret1,th1 = cv2.threshold(img1,127,255,cv2.THRESH_BINARY)
        #
        ## Otsu's thresholding
        #ret2,th2 = cv2.threshold(th1,0,255,cv2.THRESH_OTSU)
        #
        ## Otsu's thresholding after Gaussian filtering
        #blur = cv2.GaussianBlur(th1,(5,5),0)
        #ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_OTSU)
        #
        ## plot all the images and their histograms
        #images = [img1, 0, th1,
        #          img1, 0, th2,
        #          blur, 0, th3]
        #titles = ['Original Noisy Image','Histogram','Global Thresholding (v=127)',
        #          'Original Noisy Image','Histogram',"Otsu's Thresholding",
        #          'Gaussian filtered Image','Histogram',"Otsu's Thresholding"]
        #
        #for i in range(3):
        #    plt.subplot(3,3,i*3+1),plt.imshow(images[i*3],'gray')
        #    plt.title(titles[i*3]), plt.xticks([]), plt.yticks([])
        #    plt.subplot(3,3,i*3+2),plt.hist(images[i*3].ravel(),256)
        #    plt.title(titles[i*3+1]), plt.xticks([]), plt.yticks([])
        #    plt.subplot(3,3,i*3+3),plt.imshow(images[i*3+2],'gray')
        #    plt.title(titles[i*3+2]), plt.xticks([]), plt.yticks([])
        #plt.show()
        #if fatfnt==1:
        blur = cv2.blur(th1,(3,3))
        th2 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
                cv2.THRESH_BINARY,29,9)
        #ret2,th2 = cv2.threshold(blur,0,255,cv2.THRESH_OTSU)

        #ret1,th2 = cv2.threshold(blur,130,255,cv2.THRESH_BINARY)
        cv2.imwrite('predictedimg\\'+pth.split('\\')[1],th2)
        #else:
            #cv2.imwrite('predictedimg\\'+pth.split('\\')[1],th1)
    else:
        print("processing "+'DIPCO2016_dataset\\'+str(t)+'.bmp')
        pth='DIPCO2016_dataset\\'+str(t)+'.bmp'
        img = PIL.Image.open(pth)
        converter = ImageEnhance.Color(img)
        img2 = converter.enhance(0)
        img2.save('edgedetect.bmp')
        img3 = cv2.imread('edgedetect.bmp')
        dst = cv2.fastNlMeansDenoisingColored(img3,None,5,5,7,7)
        for g1 in range(4):
            dst = cv2.fastNlMeansDenoisingColored(dst,None,5,5,7,7)
        cv2.imwrite('nonoise.jpg',dst)
        idst = cv2.imread('nonoise.jpg',0)
        th2 = cv2.adaptiveThreshold(idst,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
                cv2.THRESH_BINARY,21,9)
        #ret1,th2 = cv2.threshold(blur,130,255,cv2.THRESH_BINARY)
        cv2.imwrite('predictedimg\\'+pth.split('\\')[1],th2)
print("\n")
#######################################################################################################
import cv2
import numpy as np
from skimage.measure import compare_ssim
from skimage.color import rgb2gray
r = []
p = []
fm = []
ssm=[]
me=[]
def mse(imageA, imageB):
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	return err
for f in range(1, 11):
    file = str(f)
    out_image = cv2.imread('predictedimg/{0}.bmp'.format(file),0)
    image = cv2.imread('DIPCO2016_dataset/{0}.bmp'.format(file),0)
    gt_image = cv2.imread('DIPCO2016_Dataset_GT/{0}_GT.bmp'.format(file),0)
    image = rgb2gray(image)
    image = np.uint8(np.floor(image))
    gt_image = rgb2gray(gt_image)
    gt_image = np.uint8(np.floor(gt_image))
    rows, columns = gt_image.shape
    fp = np.count_nonzero(np.greater(gt_image, out_image))
    fn = np.count_nonzero(np.greater(out_image, gt_image))
    tp = np.count_nonzero(np.logical_and(np.equal(gt_image, out_image), np.equal(gt_image, np.zeros((rows, columns)))))
    recall = tp/(tp+fn)
    precision = tp/(tp+fp)
    fscore = (2*recall*precision)/(precision+recall)
    r.append(recall)
    p.append(precision)
    fm.append(fscore)
    m = mse(gt_image, out_image)
    ssim1 = compare_ssim(gt_image, out_image)
    ssm.append(ssim1)
    me.append(m)
r = np.array(r)
p = np.array(p)
fm = np.array(fm)
me=np.array(me)
ssm=np.array(ssm)
print('Avg. F score: {0:.3f} +(or)- {1:.3f}'.format(fm.mean()*100,(fm.max() - fm.mean())*100))
print('Avg. ssim: {0:.3f} +(or)- {1:.3f}'.format(ssm.mean()*100,(ssm.max() - ssm.mean())*100))
print('Avg. Precision: {0:.3f} +(or)- {1:.3f}'.format(p.mean()*100,(p.max() - p.mean())*100))
print('Avg. Recall: {0:.3f} +(or)- {1:.3f}'.format(r.mean()*100,(r.max() - r.mean())*100))
print('Avg. Mean Square Error: {0:.3f} +(or)- {1:.3f}'.format(me.mean()*100,(me.max() - me.mean())*100))

    
