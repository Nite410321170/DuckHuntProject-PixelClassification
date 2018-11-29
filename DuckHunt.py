##############################################
#
#	TuWorld Slader
#
#	410321170
#
#	Pattern Recognition Assignment 1
#
#############################################

import cv2
import numpy as np
import sys
import math
import time

## Files
##########

duck_training_file = 'ducksFULL.jpg'
noduck_training_file = 'noducksFULL2.jpg'
test_file = 'full_duck.jpg'
output_file = "attemptFINAL.jpg"


##  Functions
##############

#Status bar function
# As suggested by Rom Ruben 
#(see: http://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console/27871113#comment50529068_27871113)
# Altered by me for use by me
def progress(count, total, status=''):
    bar_len = 40
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = ('=' * filled_len + '-' * (bar_len - filled_len))
    sys.stdout.write("\r[%s] %s%s  | %s" % (bar, percents, '%', status))
    sys.stdout.flush()

#Matrix Mean Function: Accepts a set of 3x1 vectors and finds the mean
def matrixMean(D):
	MeanV = np.matrix([0,0,0]).T
	
	b=1
	bmax = len(D)
	prog = "Computing mean"
	
	for y in range(len(D)):
		MeanV = MeanV + np.matrix(D[y]).T
		
		if (b % 100 == 0 or b==bmax):
			if b == bmax:
				prog = "Successfully Generated    "
			progress(b, bmax, prog)
		b+=1
	print()
	
	MeanV = MeanV / len(D)
	return MeanV
#Matrix Variance Function: 	Accepts a set of 3x1 vectors and the 
#				mean vector of that set and calculates the variance	
def matrixVariance(D, mean):
	VarV = np.matrix(np.zeros(shape=(3,3)))
	
	b=1
	bmax = len(D)
	prog = "Computing covariance"
	
	for x in range(len(D)):
		temp = np.matrix(D[x]).T - mean
		#print(temp,"\n")
		tempSqr = temp * temp.T
		#print(tempSqr)
		tempSum = VarV + tempSqr
		VarV = tempSum
		
		if (b % 100 == 0 or b==bmax):
			if b == bmax:
				prog = "Successfully Generated     "
			progress(b, bmax, prog)
		b+=1
	print()
	
	VarV = VarV / (len(D)-1)
	return VarV

#Data Collector Function: 	Adds the pixel information for each 
#				pixel in an image to a 1D list of 
#				size=(pixel length x pixel width) of the image.
def dataCollector(img):
	temp = list()
	
	b=1
	bmax = len(img) * len(img[0])
	prog = "Collecting data"
	
	for x in range(len(img)):
		for y in range(len(img[0])):
			temp.append(img[x][y])
			
			if (b % 100 == 0 or b==bmax):
				if b == bmax:
					prog = "Successfully Collected"
				progress(b, bmax, prog)
			b+=1
	print()
	
	return temp

def gaus_f(mean_Vec, variance_Vec, x_Vec):
	d = len(mean_Vec)
	sqrDet = math.sqrt(np.linalg.det(variance_Vec))
	pi = math.pi
	INV_variance_Vec = np.linalg.inv(variance_Vec)
	
	ans1 = (1 / ((math.pow(2*pi,d/2))*sqrDet))
	ans2 = math.exp(-0.5*((((x_Vec - mean_Vec).T)*INV_variance_Vec)*(x_Vec - mean_Vec)))
	return ans1*ans2



##  Reading in pics
####################

imgk2 = cv2.imread(test_file)
imgd = cv2.imread(duck_training_file)
imgnd = cv2.imread(noduck_training_file) 



##  Collecting training data
#############################

print("\n\n###        Collect training data        ###")
duck = dataCollector(imgd)
noduck = dataCollector(imgnd)



##  Finding Duck mean and covariance
#####################################

print("\n\n###    Find Duck mean and covariance     ###")
d_MeanV = matrixMean(duck)
d_VarianceV = matrixVariance(duck,d_MeanV)

print("Duck mean vector:\n",d_MeanV)
print("Duck covariance vector:\n",d_VarianceV,"\n\n")


##  Finding NoDuck mean and covariance
#######################################

print("### Finding NoDuck mean and covariance ###")
nd_MeanV = matrixMean(noduck)
nd_VarianceV = matrixVariance(noduck,nd_MeanV)

print("NoDuck mean vector:\n",nd_MeanV)
print("NoDuck covariance vector:\n",nd_VarianceV,"\n\n")



## Classifying image pixels
############################
print("###     Classifying the image pixels    ###\n")

#Sets row and col values
row = len(imgk2)
col = len(imgk2[0])


b = 1
bmax = row*col
#Time values
start = time.time()
seconds = 0
mins = 0
hrs = 0
prog = "["+str(hrs)+":"+str(mins)+":"+str(seconds)+"] duck huntin"


for x in range(row):
	for y in range(col):
		# An extra line to exclude the black areas of the image that aren't actually
		# apart of the image. 
		if(imgk2[x][y][0] != 0 or imgk2[x][y][1] != 0 or imgk2[x][y][2] != 0):
			probDuck = gaus_f(d_MeanV, d_VarianceV, np.matrix(imgk2[x][y]).T)
			probNoDuck = gaus_f(nd_MeanV, nd_VarianceV, np.matrix(imgk2[x][y]).T)
			prob = (probDuck - probNoDuck)
			
			if(prob>0):
				imgk2[x][y][0] = 250
				imgk2[x][y][1] = 250
				imgk2[x][y][2] = 250
			else:
				imgk2[x][y][0] = 0
				imgk2[x][y][1] = 0
				imgk2[x][y][2] = 0
				
		if (b % 100 == 0 or b==bmax):
			elapsedTime = int(time.time() - start)
			hrs = int(elapsedTime / 3600)
			elapsedTime = elapsedTime % 3600
			mins = int(elapsedTime / 60)
			elapsedTime = elapsedTime % 60
			seconds = elapsedTime
			prog = "["+str(hrs)+":"+str(mins)+":"+str(seconds)+"] duck huntin!"
			if b == bmax:
				prog = "Successfully Generated     "
			progress(b, bmax, prog)
		b+=1
print()		

## Printing out Results
########################
	
print("Classification time:",str(hrs)+" hrs "+str(mins)+" mins "+str(seconds),"secs")
cv2.imwrite(output_file, imgk2)
cv2.imshow("Result", imgk2)
cv2.waitKey(0)
