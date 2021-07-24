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
from models.resnet_custom import resnet50_baseline
from types import SimpleNamespace
from collections import namedtuple
import h5py
import yaml
from wsi_core.batch_process_utils import initialize_df
from utils.heatmap_utils import initialize_wsi, compute_from_patches, drawHeatmap, compute_from_patches_memory
from wsi_core.wsi_utils import sample_rois, to_percentiles
from utils.file_utils import save_hdf5
import time
parser = argparse.ArgumentParser(description='Heatmap Inference Script')
# parser.add_argument('--results_dir', type=str, default='results',
# 					help='experiment code')
# parser.add_argument('--models_exp_code', type=str, default=None,
# 					help='experiment code')

# parser.add_argument('--ext', type=str, default='jpeg')
# parser.add_argument('--fold', type=int, default=-1)
# parser.add_argument('--calc_heatmap', action='store_true', default=False)
# parser.add_argument('--use_center_shift', action='store_true', default=False)
# parser.add_argument('--use_roi', action='store_true', default=False)
# parser.add_argument('--no_blur', action='store_true', default=False)
# parser.add_argument('--no_drop_out', action='store_false', default=True)
# parser.add_argument('--model_size', type=str, choices=['small', 'big'], default='small', help='size of model')
# parser.add_argument('--use_ref_scores', action='store_true', default=False)
# parser.add_argument('--binarize', action='store_true', default=False)
# parser.add_argument('--alpha', type=float, default=0.5)
# parser.add_argument('--binary_thresh', type=float, default=-1)
# parser.add_argument('--blank_canvas', action='store_true', default=False)
# parser.add_argument('--save_orig', action='store_true', default=False)
# parser.add_argument('--use_features', action='store_true', default=False)
# parser.add_argument('--csv_path', type=str,
# 					help='experiment code')
# parser.add_argument('--vis_level', type=int, default=-1)
parser.add_argument('--save_exp_code', type=str, default=None,
 					help='experiment code')
parser.add_argument('--overlap', type=float, default=None)
parser.add_argument('--config_file', type=str, default="heatmap_config_test.yaml")
args = parser.parse_args()

def infer_single_slide(model, features, label, reverse_label_dict, k=1):
	# features = features.to(device)
	print(features.shape)
	with torch.no_grad():
		if isinstance(model, CLAM):
			logits, Y_prob, Y_hat, A, results_dict = model(features)
			Y_hat = Y_hat.item()
			A = A[Y_hat]
			A = A.view(-1, 1).cpu().numpy()
		else:
			raise NotImplementedError

	print('Y_hat: {}, Y: {}, Y_prob: {}'.format(reverse_label_dict[Y_hat], label, ["{:.4f}".format(p) for p in Y_prob.cpu().flatten()]))				
	probs, ids = torch.topk(Y_prob, k)
	probs = probs[-1].cpu().numpy()
	ids = ids[-1].cpu().numpy()
	#preds_str = np.array(class_labels)[ids]
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

	for key, value in config_dict.items():
		if isinstance(value, dict):
			print('\n'+key)
			for value_key, value_value in value.items():
				print (value_key + " : " + str(value_value))
		else:
			print ('\n'+key + " : " + str(value))

	decision = input('Continue? Y/N ')
	if decision in ['Y', 'y', 'Yes', 'yes']:
		pass
	elif decision in ['N', 'n', 'No', 'NO']:
		exit()
	else:
		raise NotImplementedError

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
	def_seg_params = {'seg_level': -1, 'sthresh': 150, 'mthresh': 1, 'close': 2, 'use_otsu': True, 
	'ref_downscale': 2, 'seg_downscale': 4}
	def_filter_params = {'a_t':1.0, 'a_h': 1.0, 'max_n_holes':5}
	def_vis_params = {'vis_level': -1, 'line_thickness': 50}
	def_patch_params = {'use_padding': True, 'contour_fn': 'four_pt_hard'}

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
	#if model_args.initiate_fn == 'initiate_model':
	model =  initiate_model(model_args, ckpt_path)
	feature_extractor = resnet50_baseline(pretrained=True)
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

	os.makedirs(exp_args.production_save_dir, exist_ok=True)
	os.makedirs(exp_args.raw_save_dir, exist_ok=True)
	os.makedirs('heatmaps/results/', exist_ok=True)
	blocky_wsi_kwargs = {'top_left': None, 'bot_right': None, 'patch_size': patch_size, 'step_size': step_size, 
	'custom_downsample':patch_args.custom_downsample, 'level': patch_args.patch_level, 'use_center_shift': heatmap_args.use_center_shift}

	top_left = None
	bot_right = None


	for i in range(len(process_stack)):
		slide_start = time.time()
		slide_id = process_stack.loc[i, 'slide_id']

		print('\nprocessing: ', slide_id)	

		try:
			label = process_stack.loc[i, 'label']

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
			slide_path = os.path.join(data_args.data_dir, slide_id)
		elif isinstance(data_args.data_dir, dict):
			data_dir_key = process_stack.loc[i, data_args.data_dir_key]
			slide_path = os.path.join(data_args.data_dir[data_dir_key], slide_id)
		else:
			raise NotImplementedError
		
		image_names = os.listdir(slide_path)
		image_names = [img_name for img_name in image_names if data_args.slide_ext in img_name]

		seg_params = def_seg_params.copy()
		filter_params = def_filter_params.copy()
		vis_params = def_vis_params.copy()

		seg_params = load_params(process_stack.loc[i], seg_params)
		filter_params = load_params(process_stack.loc[i], filter_params)
		vis_params = load_params(process_stack.loc[i], vis_params)

		for key, val in seg_params.items():
			print('{}: {}'.format(key, val))

		for key, val in filter_params.items():
			print('{}: {}'.format(key, val))

		for key, val in vis_params.items():
			print('{}: {}'.format(key, val))

		features_path = os.path.join(r_slide_save_dir, slide_id+'.pt')
		block_map_save_path = os.path.join(r_slide_save_dir, '{}_heatmap.h5'.format(slide_id))

		init_times = np.zeros((len(image_names))) 
		feat_times = np.zeros((len(image_names))) 
		
		all_features = []
		all_coords = []
		for idx, img_name in enumerate(image_names):
			start = time.time()
			img_id = img_name.replace(data_args.slide_ext, '')
			mask_file = os.path.join(r_slide_save_dir, img_id+'_mask.pkl')
			img_path = os.path.join(slide_path, img_name)
			# Load segmentation and filter parameters
			print('\nInitializing WSI object for {}'.format(img_name))
			wsi_object = initialize_wsi(img_path, seg_mask_path=mask_file, seg_params=seg_params, filter_params=filter_params)
			print('Done!')
			
			wsi_ref_downsample = wsi_object.level_downsamples[patch_args.patch_level]
			vis_patch_size = tuple((np.array(patch_size) * np.array(wsi_ref_downsample) * patch_args.custom_downsample).astype(int))

			mask_path = os.path.join(r_slide_save_dir, '{}_mask.jpg'.format(img_id))
			if vis_params['vis_level'] < 0:
				best_level = wsi_object.wsi.get_best_level_for_downsample(32)
				vis_params['vis_level'] = best_level
			mask = wsi_object.visWSI(**vis_params)
			mask.save(mask_path)
			h5_path = os.path.join(r_slide_save_dir, img_id+'.h5')
			init_times[idx] = time.time() - start

			#if not os.path.isfile(h5_path):
			start = time.time()
			features, coords, wsi_object = compute_from_patches_memory(wsi_object=wsi_object, 
															feature_extractor=feature_extractor, 
															batch_size=exp_args.batch_size, **blocky_wsi_kwargs)
			all_features.append(features)
			all_coords.append(coords)
			del features
			del coords

			feat_times[idx] = time.time() - start						
				# file = h5py.File(h5_path, "r")
				# features = torch.tensor(file['features'][:])
				# coords = file['coords'][:]
				# torch.save(features, features_path)
				# file.close()
				# if not os.path.isfile(mask_file):
				# 	wsi_object.saveSegmentation(mask_file)


			# load features 
			# features = torch.load(features_path)

		start = time.time()
		image_ids = [img_name.replace(data_args.slide_ext, '') for img_name in image_names]
		# h5_paths = [os.path.join(r_slide_save_dir, img_id+'.h5') for img_id in image_ids]
		all_counts = np.array([len(coords) for coords in all_coords]) 
		all_features = torch.cat(all_features, dim=0)
		all_coords = np.concatenate(all_coords, axis=0)
		
		# for h5_path in h5_paths:
		# 	file = h5py.File(h5_path, "r")
		# 	features.append(file['features'][:])
		# 	coords.append(file['coords'][:])
		# 	counts.append(len(file['coords']))
		# 	file.close()

		# features = torch.tensor(np.vstack(features))
		# coords = np.vstack(coords)
		# counts = np.array(counts)
		feat_collect_time = time.time() - start

		start = time.time()
		Y_hats, Y_hats_str, Y_probs, A = infer_single_slide(model, all_features, label, reverse_label_dict, 1)
		infer_time = time.time() - start

		Y_hat = Y_hats[0]
		process_stack.loc[i, 'total_bag_size'] = len(all_features)
		del all_features

		#if not os.path.isfile(block_map_save_path):
		asset_dict = {'attention_scores': A, 'coords': all_coords}
		# block_map_save_path = save_hdf5(block_map_save_path, asset_dict, mode='w')

		# save top 3 predictions
		for c in range(min(exp_args.n_classes, 1)):
			process_stack.loc[i, 'Pred_{}'.format(c)] = Y_hats_str[c]
			process_stack.loc[i, 'p_{}'.format(c)] = Y_probs[c]

		if data_args.process_list is not None:
			process_stack.to_csv('heatmaps/results/{}.csv'.format(data_args.process_list.replace('.csv', '')), index=False)
		else:
			process_stack.to_csv('heatmaps/results/{}.csv'.format(exp_args.save_exp_code), index=False)
		
		'''
		file = h5py.File(block_map_save_path, 'r')
		dset = file['attention_scores']
		coord_dset = file['coords']
		scores = dset[:]
		coords = coord_dset[:]
		file.close()
		'''
		# samples = sample_args.samples
		# for sample in samples:
		# 	if sample['sample']:
		# 		# sample_save_dir =  os.path.join(exp_args.save_dir, exp_args.save_exp_code, site, slide_id, sample['name'])reverse_label_dict[label]
		# 		sample_save_dir =  os.path.join(exp_args.production_save_dir, exp_args.save_exp_code, 'sampled_patches', grouping, sample['name'])
		# 		os.makedirs(sample_save_dir, exist_ok=True)
		# 		print('sampling {}'.format(sample['name']))
		# 		sample_results = sample_rois(scores, coords, k=sample['k'], mode=sample['mode'], seed=sample['seed'], 
		# 			score_start=sample.get('score_start', 0), score_end=sample.get('score_end', 1))
		# 		for idx, (s_coord, s_score) in enumerate(zip(sample_results['sampled_coords'], sample_results['sampled_scores'])):
		# 			print('coord: {} score: {:.3f}'.format(s_coord, s_score))
		# 			patch = wsi_object.wsi.read_region(tuple(s_coord), patch_args.patch_level, (patch_args.patch_size, patch_args.patch_size)).convert('RGB')
		# 			patch.save(os.path.join(sample_save_dir, '{}_{}_x_{}_y_{}_a_{:.3f}.png'.format(idx, slide_id, s_coord[0], s_coord[1], s_score)))

		# wsi_kwargs = {'top_left': top_left, 'bot_right': bot_right, 'patch_size': patch_size, 'step_size': step_size, 
		# 'custom_downsample':patch_args.custom_downsample, 'level': patch_args.patch_level, 'use_center_shift': heatmap_args.use_center_shift}

		# heatmap_save_name = '{}_heatmap_{}'.format(slide_id)
		# if os.path.isfile(os.path.join(r_slide_save_dir, heatmap_save_name)):
		# 	pass
		# else:
		# 	heatmap = drawHeatmap(scores, coords, slide_path, wsi_object=wsi_object, cmap='coolwarm', alpha=0.65, use_holes=True, binarize=False, vis_level=-1, blank_canvas=False,
		# 					thresh=0.95, patch_size = vis_patch_size, convert_to_percentiles=True, line_thickness=16)
		
		# 	heatmap.save(os.path.join(r_slide_save_dir, '{}_blockmap.png'.format(slide_id)))
		# 	del heatmap

		# save_path = os.path.join(r_slide_save_dir, '{}_{}_roi_{}.h5'.format(slide_id, patch_args.overlap, heatmap_args.use_roi))

		# if heatmap_args.use_ref_scores:
		# 	ref_scores = scores
		# else:
		# 	ref_scores = None
		
		# if heatmap_args.calc_heatmap:
		# 	compute_from_patches(wsi_object=wsi_object, clam_pred=Y_hat, model=model, feature_extractor=feature_extractor, batch_size=exp_args.batch_size, **wsi_kwargs, 
		# 						 attn_save_path=save_path, mask_path=mask_path, ref_scores=ref_scores)

		# if not os.path.isfile(save_path):
		# 	print('heatmap {} not found'.format(save_path))
		# 	if heatmap_args.use_roi:
		# 		save_path_full = os.path.join(r_slide_save_dir, '{}_{}_roi_False.h5'.format(slide_id, patch_args.overlap))
		# 		print('found heatmap for whole slide')
		# 		save_path = save_path_full
		# 	else:
		# 		continue

		# file = h5py.File(save_path, 'r')
		# dset = file['attention_scores']
		# coord_dset = file['coords']
		# scores = dset[:]
		# coords = coord_dset[:]
		# file.close()

		a_path = None
		scores = A
		scores = to_percentiles(scores)

		heatmap_vis_args = {'convert_to_percentiles': True, 'vis_level': heatmap_args.vis_level, 'blur': heatmap_args.blur, 'custom_downsample': heatmap_args.custom_downsample}
		if heatmap_args.use_ref_scores:
			heatmap_vis_args['convert_to_percentiles'] = False
		cum_counts = np.append(np.array([0]), np.cumsum(all_counts))
		
		heatmap_times = np.zeros((len(image_ids)))
		for idx, img_id in enumerate(image_ids):
			start = time.time()
			img_path = os.path.join(slide_path, img_name)
			mask_file = os.path.join(r_slide_save_dir, img_id+'_mask.pkl')
			# Load segmentation and filter parameters
			print('\nDrawing heatmap for {}'.format(img_name))
			wsi_object = initialize_wsi(img_path, seg_mask_path=mask_file, seg_params=seg_params, filter_params=filter_params)
			heatmap_save_name = '{}_{}_blur_{}_rs_{}_bc_{}_a_{}_l_{}_bi_{}_{}.{}'.format(img_id, float(patch_args.overlap),
																							int(heatmap_args.blur), 
																							int(heatmap_args.use_ref_scores), int(heatmap_args.blank_canvas), 
																							float(heatmap_args.alpha), int(heatmap_args.vis_level), 
																							int(heatmap_args.binarize), float(heatmap_args.binary_thresh), heatmap_args.save_ext)


			#if not os.path.isfile(os.path.join(p_slide_save_dir, heatmap_save_name)):
			img_scores = scores[cum_counts[idx]:cum_counts[idx+1]]  
			img_coords = all_coords[cum_counts[idx]:cum_counts[idx+1]]                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
			heatmap = drawHeatmap(img_scores, img_coords, img_path, wsi_object=wsi_object,  
									  cmap='coolwarm', alpha=heatmap_args.alpha, annotation=None, **heatmap_vis_args, 
									  binarize=heatmap_args.binarize, 
									  blank_canvas=heatmap_args.blank_canvas,
									  thresh=heatmap_args.binary_thresh,  patch_size = vis_patch_size, line_thickness=28, 
									  overlap=patch_args.overlap, 
									  top_left=top_left, bot_right = bot_right)
			if heatmap_args.save_ext == 'jpg':
				heatmap.save(os.path.join(p_slide_save_dir, heatmap_save_name), quality=100)
			else:
				heatmap.save(os.path.join(p_slide_save_dir, heatmap_save_name))
			heatmap_times[idx] = time.time() - start
		# if heatmap_args.save_orig:
		# 	if heatmap_args.vis_level >= 0:
		# 		vis_level = heatmap_args.vis_level
		# 	else:
		# 		vis_level = vis_params['vis_level']
		# 	heatmap_save_name = '{}_orig_{}.{}'.format(slide_id,int(vis_level), heatmap_args.save_ext)
		# 	if os.path.isfile(os.path.join(p_slide_save_dir, heatmap_save_name)):
		# 		pass
		# 	else:
		# 		# wsi_object = WholeSlideImage(slide_path)
		# 		heatmap = wsi_object.visWSI(vis_level=vis_level, view_slide_only=True, custom_downsample=heatmap_args.custom_downsample)
		# 		if heatmap_args.save_ext == 'jpg':
		# 			heatmap.save(os.path.join(p_slide_save_dir, heatmap_save_name), quality=100)
		# 		else:
		# 			heatmap.save(os.path.join(p_slide_save_dir, heatmap_save_name))
		print('overall time for slide {}: {:.3f}s'.format(slide_id, time.time() - slide_start))
		print('initialize, avg time per ROI: {:.3f}s'.format(np.mean(init_times)))
		print('patching and feature extraction, avg time per ROI: {:.3f}s'.format(np.mean(feat_times)))
		print('heatmap generation, avg time per ROI: {:.3f}s'.format(np.mean(heatmap_times)))
		print('inference time: {:.3f}s'.format(infer_time))

	with open(os.path.join(exp_args.raw_save_dir, exp_args.save_exp_code, 'config.yaml'), 'w') as outfile:
		yaml.dump(config_dict, outfile, default_flow_style=False)



