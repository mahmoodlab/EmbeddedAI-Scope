import math
import os
import time
import multiprocessing as mp
import cv2
import matplotlib.pyplot as plt
import numpy as np
import openslide
from PIL import Image, ImageOps
import pdb
import h5py
import math
import itertools
from wsi_core.WholeSlideImage import WholeSlideImage

class CellPhoneImage(WholeSlideImage):
	def __init__(self, path):
		super(CellPhoneImage, self).__init__(path)


	def segmentTissue_cutout(self, seg_level=0, sthresh=150, mthresh=7, close=0, use_otsu=True, 
							filter_params={'a_t':1, 'a_h': 1, 'max_n_holes':3}, ref_patch_size=512, ref_downscale=2, seg_downscale=1, invert=True):
		"""
			Segment the tissue via HSV -> Median thresholding -> Binary threshold
		"""
		
		def _filter_contours(contours, hierarchy, filter_params):
			"""
				Filter contours by: area.
			"""
			filtered = []

			# find foreground contours (parent == -1)
			hierarchy_1 = np.flatnonzero(hierarchy[:,1] == -1)

			for cont_idx in hierarchy_1:
				# actual contour
				cont = contours[cont_idx]
				# indices of holes contained in this contour (children of parent contour)
				holes = np.flatnonzero(hierarchy[:, 1] == cont_idx)
				# take contour area (includes holes)
				a = cv2.contourArea(cont)
				# calculate the contour area of each hole
				hole_areas = [cv2.contourArea(contours[hole_idx]) for hole_idx in holes]
				# actual area of foreground contour region
				a = a - np.array(hole_areas).sum()
				if a == 0: continue
				if tuple((filter_params['a_t'],)) < tuple((a,)): 
					filtered.append(cont_idx)

			all_holes = []
			for parent in filtered:
				all_holes.append(np.flatnonzero(hierarchy[:, 1] == parent))

			foreground_contours = [contours[cont_idx] for cont_idx in filtered]
			
			hole_contours = []

			for hole_ids in all_holes:
				unfiltered_holes = [contours[idx] for idx in hole_ids ]
				unfilered_holes = sorted(unfiltered_holes, key=cv2.contourArea, reverse=True)
				unfilered_holes = unfilered_holes[:filter_params['max_n_holes']]
				filtered_holes = []
				
				for hole in unfilered_holes:
					if cv2.contourArea(hole) > filter_params['a_h']:
						filtered_holes.append(hole)

				hole_contours.append(filtered_holes)

			return foreground_contours, hole_contours
		
		w, h = self.level_dim[seg_level]
		seg_downscale = (seg_downscale, seg_downscale)
		ref_downscale = (ref_downscale, ref_downscale)
		img = self.wsi.read_region((0,0), seg_level, self.level_dim[seg_level]).convert('L').resize((int(w/seg_downscale[0]), int(h/seg_downscale[1])))

		if invert:
			img = ImageOps.invert(img)

		img = np.array(img)
		img_med = cv2.medianBlur(img, mthresh)  # Apply median blurring
		
		# Thresholding
		if use_otsu:
			_, img_otsu = cv2.threshold(img_med, 0, 255, cv2.THRESH_OTSU+cv2.THRESH_BINARY)
		else:
			_, img_otsu = cv2.threshold(img_med, sthresh, 255, cv2.THRESH_BINARY)

		# Invert mask
		img_otsu = ~img_otsu

		# Morphological closing
		if close > 0:
			kernel = np.ones((close, close), np.uint8)
			img_otsu = cv2.morphologyEx(img_otsu, cv2.MORPH_CLOSE, kernel)                 

		# scale = self.level_downsamples[seg_level]
		scaled_ref_patch_area = int(ref_patch_size**2 / (ref_downscale[0] * ref_downscale[1] * seg_downscale[0] * seg_downscale[1]))
		filter_params = filter_params.copy()
		filter_params['a_t'] = filter_params['a_t'] * scaled_ref_patch_area 
		filter_params['a_h'] = filter_params['a_h'] * scaled_ref_patch_area

		# Find and filter contours
		contours, hierarchy = cv2.findContours(img_otsu, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE) # Find contours 
		hierarchy = np.squeeze(hierarchy, axis=(0,))[:,2:]
		if filter_params: foreground_contours, hole_contours = _filter_contours(contours, hierarchy, filter_params)  # Necessary for filtering out artifacts
																 
		self.contours_tissue = self.scaleContourDim(foreground_contours, seg_downscale)
		self.holes_tissue = self.scaleHolesDim(hole_contours, seg_downscale)
		# self.contours_tissue = foreground_contours
		# self.holes_tissue = hole_contours
		self.seg_level = seg_level

	def segmentTissue(self, seg_level=0, sthresh=150, mthresh=7, close=0, use_otsu=True, 
							filter_params={'a_t':1, 'a_h': 1, 'max_n_holes':3}, ref_patch_size=128, ref_downscale=2, 
							seg_downscale=1, invert=False, exclude_ids=[], keep_ids=[]):
		"""
			Segment the tissue via HSV -> Median thresholding -> Binary threshold
		"""
		
		def _filter_contours(contours, hierarchy, filter_params):
			"""
				Filter contours by: area.
			"""
			filtered = []

			# find foreground contours (parent == -1)
			hierarchy_1 = np.flatnonzero(hierarchy[:,1] == -1)

			for cont_idx in hierarchy_1:
				# actual contour
				cont = contours[cont_idx]
				# indices of holes contained in this contour (children of parent contour)
				holes = np.flatnonzero(hierarchy[:, 1] == cont_idx)
				# take contour area (includes holes)
				a = cv2.contourArea(cont)
				# calculate the contour area of each hole
				hole_areas = [cv2.contourArea(contours[hole_idx]) for hole_idx in holes]
				# actual area of foreground contour region
				a = a - np.array(hole_areas).sum()
				if a == 0: continue
				if tuple((filter_params['a_t'],)) < tuple((a,)): 
					filtered.append(cont_idx)

			all_holes = []
			for parent in filtered:
				all_holes.append(np.flatnonzero(hierarchy[:, 1] == parent))

			foreground_contours = [contours[cont_idx] for cont_idx in filtered]
			
			hole_contours = []

			for hole_ids in all_holes:
				unfiltered_holes = [contours[idx] for idx in hole_ids ]
				unfilered_holes = sorted(unfiltered_holes, key=cv2.contourArea, reverse=True)
				unfilered_holes = unfilered_holes[:filter_params['max_n_holes']]
				filtered_holes = []
				
				for hole in unfilered_holes:
					if cv2.contourArea(hole) > filter_params['a_h']:
						filtered_holes.append(hole)

				hole_contours.append(filtered_holes)

			return foreground_contours, hole_contours
		
		w, h = self.level_dim[seg_level]
		seg_downscale = (seg_downscale, seg_downscale)
		ref_downscale = (ref_downscale, ref_downscale)
		# img = self.wsi.read_region((0,0), seg_level, self.level_dim[seg_level]).convert('L').resize((int(w/seg_downscale[0]), int(h/seg_downscale[1])))
		img = np.array(self.wsi.read_region((0,0), seg_level, self.level_dim[seg_level]).convert('RGB').resize((int(w/seg_downscale[0]), int(h/seg_downscale[1]))))
		img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)  # Convert to HSV space
		img_med = cv2.medianBlur(img_hsv[:,:,1], mthresh)  # Apply median blurring
		
		# Thresholding
		if use_otsu:
			_, img_otsu = cv2.threshold(img_med, 0, 255, cv2.THRESH_OTSU+cv2.THRESH_BINARY)
		else:
			_, img_otsu = cv2.threshold(img_med, sthresh, 255, cv2.THRESH_BINARY)

		# Morphological closing
		if close > 0:
			kernel = np.ones((close, close), np.uint8)
			img_otsu = cv2.morphologyEx(img_otsu, cv2.MORPH_CLOSE, kernel)                 

		# scale = self.level_downsamples[seg_level]
		scaled_ref_patch_area = int(ref_patch_size**2 / (ref_downscale[0] * ref_downscale[1] * seg_downscale[0] * seg_downscale[1]))
		filter_params = filter_params.copy()
		filter_params['a_t'] = filter_params['a_t'] * scaled_ref_patch_area 
		filter_params['a_h'] = filter_params['a_h'] * scaled_ref_patch_area

		# Find and filter contours
		contours, hierarchy = cv2.findContours(img_otsu, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE) # Find contours 
		hierarchy = np.squeeze(hierarchy, axis=(0,))[:,2:]
		if filter_params: foreground_contours, hole_contours = _filter_contours(contours, hierarchy, filter_params)  # Necessary for filtering out artifacts
																 
		self.contours_tissue = self.scaleContourDim(foreground_contours, seg_downscale)
		self.holes_tissue = self.scaleHolesDim(hole_contours, seg_downscale)

		if len(keep_ids) > 0:
			contour_ids = set(keep_ids) - set(exclude_ids)
		else:
			contour_ids = set(np.arange(len(self.contours_tissue))) - set(exclude_ids)

		self.contours_tissue = [self.contours_tissue[i] for i in contour_ids]
		self.holes_tissue = [self.holes_tissue[i] for i in contour_ids]
		# self.contours_tissue = foreground_contours
		# self.holes_tissue = hole_contours
		self.seg_level = seg_level

	def visWSI(self, vis_level=0, vis_scale=(0.5, 0.5), color = (0,255,0), hole_color = (0,0,255), annot_color=(255,0,0), 
					line_thickness=12, crop_window=None):
		img = np.array(self.wsi.read_region((0,0), vis_level, self.level_dim[vis_level]).convert("RGB"))
		downsample = self.level_downsamples[vis_level]
		scale = [1/downsample[0], 1/downsample[1]] # Scaling from 0 to desired level
		line_thickness = int(line_thickness * math.sqrt(scale[0] * scale[1]))
		if self.contours_tissue is not None:
			cv2.drawContours(img, self.scaleContourDim(self.contours_tissue, scale), 
							 -1, color, line_thickness, lineType=cv2.LINE_8)

			for holes in self.holes_tissue:
				cv2.drawContours(img, self.scaleContourDim(holes, scale), 
								 -1, hole_color, line_thickness, lineType=cv2.LINE_8)
		
		if self.contours_tumor is not None:
			cv2.drawContours(img, self.scaleContourDim(self.contours_tumor, scale), 
							 -1, annot_color, line_thickness, lineType=cv2.LINE_8)
		
		img = Image.fromarray(img)
		if crop_window is not None:
			top, left, bot, right = crop_window
			left = int(left * scale[0])
			right = int(right * scale[0])
			top =  int(top * scale[1])
			bot = int(bot * scale[1])
			crop_window = (top, left, bot, right)
			img = img.crop(crop_window)
		
		w, h = img.size
		if not np.all(np.array(vis_scale)==1):
			img = img.resize((int(w*vis_scale[0]), int(h*vis_scale[1])))
	   
		return img
