import cv2
import math
import numpy as np
from pywt import dwt2
from scipy.stats import kurtosis, skew

from skimage.feature import greycomatrix,greycoprops
# no of training images
n=int(input("please enter number of training images:"))

# extract features for training images
# features extracted for each image are:
#contrast
#correlation
#energy
#homogenity
#mean
#standard deviation
#enropy
#variance
#smoothness
#skewness
#kurtosis
def contrast(img):
    max_value=np.max(img)
    min_value=np.min(img)
    average=(float(max_value)+float(min_value))/2
    r=img.shape
    sum1=0
    for i in range(0,r[0]):
        for j in range(0,r[1]):
            sum1=sum1+((img[i][j]-average)**2)
    n=r[0]*r[1]
    sum1=sum1
    out=np.sqrt(sum1)
    return out
def autocorrelation(img):
    r=img.shape
    sum1=0
    for i in range(0,r[0]):
        for j in range(0,r[1]):
          sum1=sum1+(img[i][j]**2)
    n=r[0]*r[1]      
    out=float(sum1)/n
    return out
           
def energy(img):
	_, (cH, cV, cD) = dwt2(img.T, 'db1')
	Energy = (cH**2 + cV**2 + cD**2).sum()/img.size
	
	return Energy
            
def mean(img):
    img=np.array(img)	
    return float(img.sum()/img.size)
def variance(img):
    max_value=np.max(img)
    min_value=np.min(img)
    average=(float(max_vaue)+float(min_value))/2
    r=img.shape
    sum1=0
    for i in range(0,r[0]):
        for j in range(0,r[1]):
            sum1=sum1+((img[i][j]-average)**2)
    n=r[0]*r[1]
    out=sum1/n
    return out 
def graycm(img):
	glcm=greycomatrix(img,[1],[0,np.pi/2],symmetric=True,normed=True)
	print("glcm",glcm.shape,type(glcm))
	glcm=glcm.reshape(glcm.shape[0]*glcm.shape[3],glcm.shape[1])
	
	return glcm
def cont(img):
	#print("glcm",glcm.shape,type(glcm))
	glcm=graycm(img)
	sum1=0
	for i in range(glcm.shape[0]):
		for j in range(glcm.shape[1]):
			sum1=sum1+(pow((i-j),2)*glcm[i][j])
	return sum1       
def homogenity(img):
	#print("glcm",glcm.shape,type(glcm))
	glcm=graycm(img)
	sum1=0
	for i in range(glcm.shape[0]):
		for j in range(glcm.shape[1]):
			sum1=sum1+float(glcm[i][j]/(1+pow((i-j),2)))
	return sum1
def entropy(img):
	#print("glcm",glcm.shape,type(glcm))
	glcm=graycm(img)
	sum1=0
	for i in range(glcm.shape[0]):
		for j in range(glcm.shape[1]):
			x=math.log(abs(glcm[i][j])+0.0001,10)
			sum1=sum1+(glcm[i][j]*x)
	return -(sum1)
def skewness(img):
	img=img.flatten()
	return skew(img)
def kurt(img):
	img=img.flatten()
	return kurtosis(img)
           
            


if __name__ == '__main__':
		training_features=[]
		for i in range(1,n+1):
		# reading images
			img_features=[]
			img=cv2.imread('Day %d.JPG' % i ,0) #gray scale image reading
			print(img.shape)
			# applying guassian filter to image 
			blur_img = cv2.GaussianBlur(img,(5,5),0) 
			resized_img=cv2.resize(blur_img,(270,270)) # resizing to reduce number of computations
			#cv2.imwrite('resized_img %d.jpg' % i,resized_img)
			# caluclating properties of each image
			print("contrast:\n",contrast(resized_img))
			print("cont:\n",cont(resized_img))
			print("skew:\n",skewness(resized_img))
			print("kurtosis:\n",kurt(resized_img))
			print("correlation:\n",autocorrelation(resized_img))
			print("energy:\n",energy(resized_img))
			print("mean:\n",mean(resized_img))
			print("homogenity:\n",homogenity(resized_img))
			print("entropy:\n",entropy(resized_img))
			img_features.append(contrast(resized_img))
			img_features.append(autocorrelation(resized_img))
			img_features.append(energy(resized_img))

			img_features.append(homogenity(resized_img))
			img_features.append(mean(resized_img))
			img_features.append(standard_deviation(resized_img))
			img_features.append(entropy(resized_img))
			img_features.append(variance(resized_img))
	#		img_features.append(smoothness(resized_img))
			img_features.append(skewness(resized_img))
			img_features.append(kurtosis(resized_img))
		training_features.append(img_features)
	
