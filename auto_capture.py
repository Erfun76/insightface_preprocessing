import mediapipe as mp
import cv2
import numpy as np
import math

# calculate camera metrics
class CameraMetric:
	def __init__(self):
		# initialize mediapipe
		mp_selfie_segmentation = mp.solutions.selfie_segmentation
		self.selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)
	# brightness
	# Average pixels, then transform to "perceived brightness".
	def calc_brightness(self, img):
		mask = self.__mask_person(img)
		mask = np.logical_not(mask)
		masked_img = np.ma.masked_array(img, mask=mask)
		b, g, r = np.mean(masked_img[:,:,0]), np.mean(masked_img[:,:,1]), np.mean(masked_img[:,:,2])
		return math.sqrt(0.241*(r**2) + 0.691*(g**2) + 0.068*(b**2))

	# contrast
	def calc_contrast(self, img):
		mask = self.__mask_person(img)
		masked_img = np.ma.masked_array(img, mask=mask)
		img_grey = np.mean(masked_img, axis=2)
		return img_grey.std()
	
	# sharpness
	# using laplacian filter
	def calc_sharpness(self, img):
		# mask = self.__mask_person(img)
		# masked_img = img*mask

		return cv2.Laplacian(img, cv2.CV_64F).var()
	
	def calc_resolution(self, resolution):
		return resolution[0]*resolution[1]
	
	def __mask_person(self, frame):
		RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		# get the result
		results = self.selfie_segmentation.process(RGB)
		# extract segmented mask
		return np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.5