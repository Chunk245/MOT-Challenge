import numpy as np
import torch
import torch.nn.functional as F

import motmetrics as mm
from torchvision.ops.boxes import clip_boxes_to_image, nms

from filterpy.kalman import KalmanFilter
from .utils import nms, convert_bbox_to_z, convert_x_to_bbox

class Tracker:
	"""The main tracking file, here is where magic happens."""

	def __init__(self, obj_detect):
		self.obj_detect = obj_detect

		self.tracks = []
		self.track_num = 0
		self.im_index = 0
		self.results = {}

		self.mot_accum = None

	def reset(self, hard=True):
		self.tracks = []

		if hard:
			self.track_num = 0
			self.results = {}
			self.im_index = 0
			self.unmatched_tracks = []

	def add(self, new_boxes, new_features, new_scores):
		"""Initializes new Track objects and saves them."""
		num_new = len(new_boxes)
		for i in range(num_new):
			self.tracks.append(Track(
				new_boxes[i],
                new_features[i],
				new_scores[i],
				self.track_num + i
			))
		self.track_num += num_new

	def get_pos(self):
		"""Get the positions of all active tracks."""
		if len(self.tracks) == 1:
			box = self.tracks[0].box
		elif len(self.tracks) > 1:
			box = torch.stack([t.box for t in self.tracks], 0)
		else:
			box = torch.zeros(0).cuda()
		return box

	def data_association(self, image, boxes, scores):
		self.tracks = []
		self.add(boxes, scores)

	def step(self, frame):
		"""This function should be called every timestep to perform tracking with a blob
		containing the image information.
		"""
		# object detection
		boxes, scores = self.obj_detect.detect(frame['img'])
        
		pred = {}
		pred['boxes'] = boxes
		pred['scores'] = scores
		pred = nms([pred])
		boxes = pred[0]['boxes']
		scores = pred[0]['scores']
        
		self.data_association(frame['img'], boxes, scores)

		# results
		for t in self.tracks:
			if t.id not in self.results.keys():
				self.results[t.id] = {}
			self.results[t.id][self.im_index] = np.concatenate([t.box.cpu().numpy(), np.array([t.score])])

		self.im_index += 1

	def get_results(self):
		return self.results


class Track(object):
	"""This class contains all necessary for every individual track."""

	def __init__(self, box, feature, score, track_id):
		self.id = track_id
		self.box = box
		self.features = [feature]
		self.score = score
        
		self.kf = KalmanFilter(dim_x=7, dim_z=4) 
		self.kf.F = np.eye(7) + np.eye(7, k=4)
		self.kf.H = np.eye(4,7)

		self.kf.R[2:,2:] *= 10.
		self.kf.P[4:,4:] *= 1000.
		self.kf.P *= 10.
		self.kf.Q[-1,-1] *= 0.01
		self.kf.Q[4:,4:] *= 0.01
		self.kf.x[:4] = convert_bbox_to_z(box)

	def get_feature(self):
		return torch.stack(self.features).mean(0)

	def add_feature(self, feature):
		if len(self.features) >= 3:
			self.features.pop(0)
		self.features.append(feature)
		return        
    
	def predict(self):
        
		if((self.kf.x[6]+self.kf.x[2])<=0):
			self.kf.x[6] *= 0.0
		self.kf.predict()
		return convert_x_to_bbox(self.kf.x)

	def get_state(self):
		return convert_x_to_bbox(self.kf.x)

