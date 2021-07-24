from __future__ import print_function

import numpy as np

import argparse

import torch
import torch.nn as nn
import pdb
import os
import pandas as pd
from utils.utils import *
from math import floor
from utils.eval_utils import initiate_model as initiate_model
from models.model_attention_mil import MIL_Attention_fc
from models.model_clam import CLAM
from models.model_mil import MIL_fc
from models.resnet_custom import resnet50_baseline
from models.ghostnet import ghostnet_baseline
from types import SimpleNamespace
from collections import namedtuple
import h5py
import yaml
from wsi_core.batch_process_utils import initialize_df
from utils.heatmap_utils import initialize_wsi, compute_from_patches, drawHeatmap
from wsi_core.wsi_utils import sample_rois, to_percentiles
from utils.file_utils import save_hdf5
from wsi_core.WholeSlideImage import WholeSlideImage
import time
import timm
from sklearn.metrics import roc_auc_score, roc_curve, auc
parser = argparse.ArgumentParser(description='Heatmap Inference Script')
parser.add_argument('--save_exp_code', type=str, default=None,
 					help='experiment code')
parser.add_argument('--overlap', type=float, default=None)
parser.add_argument('--feat_save_path', type=str, default=None)
parser.add_argument('--no_overwrite', action='store_true', default=False)
parser.add_argument('--save_patches', action='store_true', default=False)
parser.add_argument('--config_file', type=str, default="heatmap_config_test.yaml")
args = parser.parse_args()

def infer_single_slide(model, features, label, reverse_label_dict, k=1):
	features = features.to(device)
	print(features.shape)
	with torch.no_grad():
		if isinstance(model, MIL_fc):
			logits, Y_prob, Y_hat, A, results_dict = model(features)
			Y_hat = Y_hat.item()
			A = A.view(-1, 1).cpu().numpy()
		elif isinstance(model, CLAM):
			logits, Y_prob, Y_hat, A, results_dict = model(features)
			Y_hat = Y_hat.item()
			A = A[Y_hat]
			A = A.view(-1, 1).cpu().numpy()
		elif isinstance(model, MIL_Attention_fc):
			logits, Y_prob, Y_hat, A, results_dict = model(features)
			Y_hat = Y_hat.item()
			A = A.view(-1, 1).cpu().numpy()
		else:
			raise NotImplementedError

	print('Y_hat: {}, Y: {}, Y_prob: {}'.format(reverse_label_dict[Y_hat], label, ["{:.4f}".format(p) for p in Y_prob.cpu().flatten()]))				
	# probs, ids = torch.topk(Y_prob, k)
	# probs = probs[-1].cpu().numpy()
	# ids = ids[-1].cpu().numpy()

	probs = Y_prob[-1].cpu().numpy()
	ids = np.arange(Y_prob.size(1))

	preds_str = np.array([reverse_label_dict[idx] for idx in ids])

	return ids, preds_str, probs, A

def load_params(df_entry, params):
	for key in params.keys():
		if key in df_entry.index:
			dtype = type(params[key])
			val = df_entry[key]
			if not np.isnan(val):
				params[key] = dtype(val)

	return params

def parse_config_dict(args, config_dict):
	if args.save_exp_code is not None:
		config_dict['exp_arguments']['save_exp_code'] = args.save_exp_code
	if args.overlap is not None:
		config_dict['patching_arguments']['overlap'] = args.overlap
	return config_dict

if __name__ == '__main__':
	config_path = os.path.join('heatmaps/configs', args.config_file)
	config_dict = yaml.safe_load(open(config_path, 'r'))
	config_dict = parse_config_dict(args, config_dict)
	pt_feat_save_path = args.feat_save_path
	no_overwrite = args.no_overwrite

	for key, value in config_dict.items():
		if isinstance(value, dict):
			print('\n'+key)
			for value_key, value_value in value.items():
				print (value_key + " : " + str(value_value))
		else:
			print ('\n'+key + " : " + str(value))

	# decision = input('Continue? Y/N ')
	# if decision in ['Y', 'y', 'Yes', 'yes']:
	# 	pass
	# elif decision in ['N', 'n', 'No', 'NO']:
	# 	exit()
	# else:
	# 	raise NotImplementedError

	args = config_dict
	patch_args = argparse.Namespace(**args['patching_arguments'])
	data_args = argparse.Namespace(**args['data_arguments'])
	model_args = args['model_arguments']
	model_args.update({'n_classes': args['exp_arguments']['n_classes']})
	model_args = argparse.Namespace(**model_args)
	exp_args = argparse.Namespace(**args['exp_arguments'])
	heatmap_args = argparse.Namespace(**args['heatmap_arguments'])
	# sample_args = argparse.Namespace(**args['sample_arguments'])

	patch_size = tuple([patch_args.patch_size for i in range(2)])
	step_size = tuple((np.array(patch_size) * (1-patch_args.overlap)).astype(int))
	print('patch_size: {} x {}, with {:.2f} overlap, step size is {} x {}'.format(patch_size[0], patch_size[1], patch_args.overlap, step_size[0], step_size[1]))

	preset = data_args.preset
	# def_seg_params = {'seg_level': -1, 'sthresh': 150, 'mthresh': 1, 'close': 2, 'use_otsu': True, 
	# 'ref_downscale': 2, 'seg_downscale': 4}
	# def_filter_params = {'a_t':1.0, 'a_h': 1.0, 'max_n_holes':5}
	# def_vis_params = {'vis_level': -1, 'line_thickness': 50}
	# def_patch_params = {'use_padding': True, 'contour_fn': 'four_pt_hard'}

	def_seg_params = {'seg_level': -1, 'sthresh': 15, 'mthresh': 11, 'close': 2, 'use_otsu': False, 
	'ref_downscale': 2, 'seg_downscale': 4, 'keep_ids': 'none', 'exclude_ids':'none'}
	def_filter_params = {'a_t':1.0, 'a_h': 1.0, 'max_n_holes':5}
	def_vis_params = {'vis_level': -1, 'line_thickness': 50}
	def_patch_params = {'use_padding': True, 'contour_fn': 'four_pt_hard'}

	# def_seg_params = {'seg_level': -1, 'sthresh': 15, 'mthresh': 11, 'close': 2, 'use_otsu': False, 
	# 				  'keep_ids': 'none', 'exclude_ids':'none'}
	# def_filter_params = {'a_t':50.0, 'a_h': 8.0, 'max_n_holes':10}
	# def_vis_params = {'vis_level': -1, 'line_thickness': 250}
	# def_patch_params = {'use_padding': True, 'contour_fn': 'four_pt_hard'}

	if preset is not None:
		preset_df = pd.read_csv(preset)
		for key in def_seg_params.keys():
			def_seg_params[key] = preset_df.loc[0, key]

		for key in def_filter_params.keys():
			def_filter_params[key] = preset_df.loc[0, key]

		for key in def_vis_params.keys():
			def_vis_params[key] = preset_df.loc[0, key]

		for key in def_patch_params.keys():
			def_patch_params[key] = preset_df.loc[0, key]


	if data_args.process_list is None:
		if isintance(data_args.data_dir, list):
			slides = []
			for data_dir in data_args.data_dir:
				slides.extend(os.listdir(data_dir))
		else:
			slides = sorted(os.listdir(data_args.data_dir))
		slides = [slide for slide in slides if data_args.slide_ext in slide]
		df = initialize_df(slides, def_seg_params, def_filter_params, def_vis_params, def_patch_params, heatmap_coords=True)
	
	else:
		df = pd.read_csv(os.path.join('heatmaps/process_lists', data_args.process_list))
		df = initialize_df(df, def_seg_params, def_filter_params, def_vis_params, def_patch_params, heatmap_coords=True)

	mask = df['process'] == 1
	process_stack = df[mask].reset_index(drop=True)
	total = len(process_stack)
	print('\nlist of slides to process: ')
	print(process_stack.head(len(process_stack)))

	print('\ninitializing model from checkpoint')
	ckpt_path = model_args.ckpt_path
	print('\nckpt path: {}'.format(ckpt_path))
	model =  initiate_model(model_args, ckpt_path)

	if model_args.enc_name == 'resnet50':
		feature_extractor = resnet50_baseline(pretrained=True)
	elif model_args.enc_name == 'ghostnet':
		feature_extractor = ghostnet_baseline(pretrained=True)
	elif model_args.enc_name == 'mobilenetv3':
		feature_extractor = timm.create_model('mobilenetv3_large_100', pretrained=True, num_classes=0)

	feature_extractor.eval()
	device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print('Done!')

	label_dict =  data_args.label_dict
	class_labels = list(label_dict.keys())
	class_encodings = list(label_dict.values())
	reverse_label_dict = {class_encodings[i]: class_labels[i] for i in range(len(class_labels))} 
	if torch.cuda.device_count() > 1:
		device_ids = list(range(torch.cuda.device_count()))
		feature_extractor = nn.DataParallel(feature_extractor, device_ids=device_ids).to('cuda:0')
	else:
		feature_extractor = feature_extractor.to(device)

	# warmup GPUs
	for i in range(3):
		x = torch.rand(exp_args.batch_size, 3, 256, 256).to(device)
		feature_extractor(x)
		del x

	os.makedirs(exp_args.production_save_dir, exist_ok=True)
	os.makedirs(exp_args.raw_save_dir, exist_ok=True)
	os.makedirs('heatmaps/results/', exist_ok=True)
	blocky_wsi_kwargs = {'top_left': None, 'bot_right': None, 'patch_size': patch_size, 'step_size': step_size, 
	'custom_downsample':patch_args.custom_downsample, 'level': patch_args.patch_level, 'use_center_shift': heatmap_args.use_center_shift}

	top_left = None
	bot_right = None

	assert 'slide_id' in process_stack.columns
	assert 'img' in process_stack.columns

	final_df = df.copy().set_index('slide_id')
	final_df = final_df[~(final_df.index.duplicated(keep='first'))]
	final_df = final_df.drop(columns=['img', 'img_id'])

	process_stack = process_stack.set_index('img')

	for i, slide_id in enumerate(process_stack.slide_id.unique()):
		# begin clock for one slide
		slide_start = time.time()

		print('\nprocessing: ', slide_id)	
		slide_df = process_stack[process_stack.slide_id == slide_id]

		try:
			label = final_df.loc[slide_id, 'label']
			# label = process_stack.loc[slide_id, 'label']

		except KeyError:
			label = 'Unspecified'

		# if data_args.slide_ext not in slide_name:
		# 	slide_name+=data_args.slide_ext
		# slide_id = slide_name.strip(data_args.slide_ext)

		if not isinstance(label, str):
			grouping = reverse_label_dict[label]
		else:
			grouping = label

		# base_dir for production files
		p_slide_save_dir = os.path.join(exp_args.production_save_dir, exp_args.save_exp_code, str(grouping), slide_id)
		os.makedirs(p_slide_save_dir, exist_ok=True)

		# base_dir for raw files 
		r_slide_save_dir = os.path.join(exp_args.raw_save_dir, exp_args.save_exp_code, str(grouping),  slide_id)
		os.makedirs(r_slide_save_dir, exist_ok=True)

		# fetch file directory for slide
		if isinstance(data_args.data_dir, str):
			slide_path = os.path.join(data_args.data_dir)
		elif isinstance(data_args.data_dir, dict):
			data_dir_key = process_stack.loc[i, data_args.data_dir_key]
			slide_path = os.path.join(data_args.data_dir[data_dir_key])
		else:
			raise NotImplementedError

		features_path = os.path.join(r_slide_save_dir, slide_id+'.pt')
		
		init_times = np.zeros((len(slide_df))) 
		feat_times = np.zeros((len(slide_df))) 

		image_names = slide_df.index.values

		for idx, img_name in enumerate(image_names):
			start = time.time() # start clock for 1 image
			seg_params = def_seg_params.copy()
			filter_params = def_filter_params.copy()
			vis_params = def_vis_params.copy()

			seg_params = load_params(process_stack.loc[img_name], seg_params)
			filter_params = load_params(process_stack.loc[img_name], filter_params)
			vis_params = load_params(process_stack.loc[img_name], vis_params)

			keep_ids = str(seg_params['keep_ids'])
			if len(keep_ids) > 0 and keep_ids != 'none':
				seg_params['keep_ids'] = np.array(keep_ids.split(',')).astype(int)
			else:
				seg_params['keep_ids'] = []

			exclude_ids = str(seg_params['exclude_ids'])
			if len(exclude_ids) > 0 and exclude_ids != 'none':
				seg_params['exclude_ids'] = np.array(exclude_ids.split(',')).astype(int)
			else:
				seg_params['exclude_ids'] = []

			for key, val in seg_params.items():
				print('{}: {}'.format(key, val))

			for key, val in filter_params.items():
				print('{}: {}'.format(key, val))

			for key, val in vis_params.items():
				print('{}: {}'.format(key, val))


			img_id = img_name.replace(data_args.slide_ext, '')
			mask_file = os.path.join(r_slide_save_dir, img_id+'_mask.pkl')
			img_path = os.path.join(slide_path, img_name)
			# Load segmentation and filter parameters
			print('\nInitializing WSI object for {}'.format(img_name))
			# wsi_object = WholeSlideImage(img_path)
			wsi_object = initialize_wsi(img_path, seg_mask_path=mask_file, seg_params=seg_params, filter_params=filter_params, segment=True)
			print('Done!')
			
			wsi_ref_downsample = wsi_object.level_downsamples[patch_args.patch_level]
			vis_patch_size = tuple((np.array(patch_size) * np.array(wsi_ref_downsample) * patch_args.custom_downsample).astype(int))
			
			###### SAVE SEGMENTATION MASKS #########
			mask_path = os.path.join(r_slide_save_dir, '{}_mask.jpg'.format(img_id))
			if vis_params['vis_level'] < 0:
				best_level = wsi_object.wsi.get_best_level_for_downsample(32)
				vis_params['vis_level'] = best_level
			mask = wsi_object.visWSI(**vis_params)
			mask.save(mask_path)

			h5_path = os.path.join(r_slide_save_dir, img_id+'.h5')
			init_times[idx] = time.time() - start # time for initialization of image

			if not os.path.isfile(h5_path) or not no_overwrite:
				start = time.time()
				_, feat_save_path = compute_from_patches(wsi_object=wsi_object, 
																model=model, 
																feature_extractor=feature_extractor, 
																batch_size=exp_args.batch_size, **blocky_wsi_kwargs, 
																attn_save_path=None, feat_save_path=h5_path, 
																ref_scores=None)

				feat_times[idx] = time.time() - start 
			else:
				feat_times[idx] = -1						


		start = time.time() # start clock for collecting features from multiple slides
		image_ids = [img_name.replace(data_args.slide_ext, '') for img_name in image_names]
		h5_paths = [os.path.join(r_slide_save_dir, img_id+'.h5') for img_id in image_ids]
		counts = []
		features = []
		coords = []
		for h5_path in h5_paths:
			file = h5py.File(h5_path, "r")
			features.append(file['features'][:])
			coords.append(file['coords'][:])
			counts.append(len(file['coords']))
			file.close()

		features = torch.tensor(np.vstack(features))
		coords = np.vstack(coords)
		counts = np.array(counts)
		feat_collect_time = time.time() - start 

		start = time.time()
		Y_hats, Y_hats_str, Y_probs, A = infer_single_slide(model, features, label, reverse_label_dict, 1)
		infer_time = time.time() - start

		Y_hat = Y_hats[0]
		final_df.loc[slide_id, 'total_bag_size'] = len(features)
		if pt_feat_save_path:
			os.makedirs(pt_feat_save_path, exist_ok=True)
			torch.save(features, os.path.join(pt_feat_save_path, slide_id+'.pt'))
		del features

		# block_map_save_path = os.path.join(r_slide_save_dir, '{}_heatmap.h5'.format(slide_id))
		# if not os.path.isfile(block_map_save_path):
		# 	asset_dict = {'attention_scores': A, 'coords': coords}
		# 	block_map_save_path = save_hdf5(block_map_save_path, asset_dict, mode='w')
		final_df.loc[slide_id, 'pred_label'] = label
		for c in range(exp_args.n_classes):
			final_df.loc[slide_id, 'Pred_{}'.format(c)] = Y_hats_str[c]
			final_df.loc[slide_id, 'p_{}'.format(c)] = Y_probs[c]

		df.to_csv('heatmaps/results/{}.csv'.format(data_args.process_list.replace('.csv', '')))

		a_path = None
		scores = A

		if model_args.model_type != 'mil':
			scores = to_percentiles(scores)

		heatmap_vis_args = {'convert_to_percentiles': False, 'vis_level': heatmap_args.vis_level, 'blur': heatmap_args.blur, 'custom_downsample': heatmap_args.custom_downsample}

		cum_counts = np.append(np.array([0]), np.cumsum(counts))
		
		heatmap_times = np.zeros((len(image_ids)))
		for idx, img_id in enumerate(image_ids):
			start = time.time()
			img_name = image_names[idx]
			img_path = os.path.join(slide_path, img_name)
			mask_file = os.path.join(r_slide_save_dir, img_id+'_mask.pkl')
			# Load segmentation and filter parameters
			print('\nDrawing heatmap for {}'.format(img_name))
			# wsi_object = WholeSlideImage(img_path)
			wsi_object = initialize_wsi(img_path, seg_mask_path=mask_file, seg_params=seg_params, filter_params=filter_params, segment=True)
			heatmap_save_name = '{}_{}_blur_{}_bc_{}_a_{}_l_{}_bi_{}_{}.{}'.format(img_id, float(patch_args.overlap),
																							int(heatmap_args.blur), 
																							int(heatmap_args.blank_canvas), 
																							float(heatmap_args.alpha), int(heatmap_args.vis_level), 
																							int(heatmap_args.binarize), float(heatmap_args.binary_thresh), heatmap_args.save_ext)


			if not no_overwrite or not os.path.isfile(os.path.join(p_slide_save_dir, heatmap_save_name)):
				img_scores = scores[cum_counts[idx]:cum_counts[idx+1]]  
				img_coords = coords[cum_counts[idx]:cum_counts[idx+1]]                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
				heatmap = drawHeatmap(img_scores, img_coords, img_path, wsi_object=wsi_object,  
										  cmap=heatmap_args.cmap, alpha=heatmap_args.alpha, annotation=None, **heatmap_vis_args, 
										  binarize=heatmap_args.binarize, 
										  blank_canvas=heatmap_args.blank_canvas,
										  thresh=heatmap_args.binary_thresh,  patch_size = vis_patch_size, line_thickness=28, 
										  overlap=patch_args.overlap, 
										  top_left=top_left, bot_right = bot_right, segment=True)
				if heatmap_args.save_ext == 'jpg':
					heatmap.save(os.path.join(p_slide_save_dir, heatmap_save_name), quality=100)
				else:
					heatmap.save(os.path.join(p_slide_save_dir, heatmap_save_name))
			heatmap_times[idx] = time.time() - start

		final_df.loc[slide_id, 'total_time'] = time.time() - slide_start
		final_df.loc[slide_id, 'init_time'] = np.mean(init_times)
		final_df.loc[slide_id, 'patch_feat_time'] = np.mean(feat_times)
		final_df.loc[slide_id, 'heatmap_time'] = np.mean(heatmap_times)
		final_df.loc[slide_id, 'feat_collect_time'] = feat_collect_time
		final_df.loc[slide_id, 'infer_time'] = infer_time
		final_df.loc[slide_id, 'n_roi'] = len(image_ids)
		final_df.to_csv('heatmaps/results/{}.csv'.format(exp_args.save_exp_code))
		
		print('overall time for slide {}: {:.3f}s'.format(slide_id, final_df.loc[slide_id, 'total_time']))
		print('initialize, avg time per ROI: {:.3f}s'.format(np.mean(init_times)))
		print('patching and feature extraction, avg time per ROI: {:.3f}s'.format(np.mean(feat_times)))
		print('collecting features across all images: {:.3f}s'.format(feat_collect_time))
		print('heatmap generation, avg time per ROI: {:.3f}s'.format(np.mean(heatmap_times)))
		print('inference time: {:.3f}s'.format(infer_time))


	labels = final_df.pred_label.map(label_dict).values
	probs = final_df.p_1.values
	roc_auc = roc_auc_score(labels, probs)
	print('auc: {:.6f}'.format(roc_auc))

	with open(os.path.join(exp_args.raw_save_dir, exp_args.save_exp_code, 'config.yaml'), 'w') as outfile:
		yaml.dump(config_dict, outfile, default_flow_style=False)



		



