
import matplotlib
import matplotlib.colors
import numpy as np
import mediapy as media
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import PIL
import io
import matplotlib.patches

def flow_to_rgb(vec, flow_mag_range=None, white_bg=False):
  height, width = vec.shape[:2]
  scaling = 50. / (height**2 + width**2)**0.5
  direction = (np.arctan2(vec[..., 0], vec[..., 1]) + np.pi) / (2 * np.pi)
  norm = np.linalg.norm(vec, axis=-1)
  if flow_mag_range is None:
    flow_mag_range = norm.min(), norm.max()
  magnitude = np.clip((norm - flow_mag_range[0]) * scaling, 0., 1.)
  if white_bg == True:
    value = np.ones_like(direction)
    hsv = np.stack([direction, magnitude, saturation], axis=-1)
  else:
    saturation = np.ones_like(direction)
    hsv = np.stack([direction, saturation , magnitude], axis=-1)
  rgb = matplotlib.colors.hsv_to_rgb(hsv)
  return rgb

def print_instance_ids(sample, ds_info, palette=None):
  if palette is None:
    palette = sns.color_palette('hls', sample["metadata"]["num_instances"])
  out = ''
  if "asset_id" in sample["instances"]:
    ids = [s.decode() for s in sample["instances"]["asset_id"]]
  else:
    labels = []
    if "size_label" in sample["instances"]:
      labels.append([ds_info.features["instances"]["size_label"].names[k]
                       for k in sample["instances"]["size_label"]])
    if "color_label" in sample["instances"]:
      labels.append([ds_info.features["instances"]["color_label"].names[k]
                       for k in sample["instances"]["color_label"]])
    labels.append([ds_info.features["instances"]["material_label"].names[k]
                       for k in sample["instances"]["material_label"]])
    labels.append([ds_info.features["instances"]["shape_label"].names[k]
                   for k in sample["instances"]["shape_label"]])
    ids = [" ".join(x) for x in zip(*labels)]

  for i, (color, asset_id) in enumerate(zip(palette, ids)):
    color_hex = '#%02x%02x%02x' % tuple(int(x*255) for x in color)
    out += f'{i}.'
    
  return html_print(out)

ds, ds_info = tfds.load("movi_f", data_dir="gs://kubric-public/tfds", with_info=True)
train_iter = iter(tfds.as_numpy(ds["train"]))

example = next(train_iter)
# minv, maxv = example["metadata"]["forward_flow_range"]
# forward_flow = example["forward_flow"] / 65535 * (maxv - minv) + minv

# minv, maxv = example["metadata"]["backward_flow_range"]
# backward_flow = example["backward_flow"] / 65535 * (maxv - minv) + minv

minv, maxv = example["metadata"]["depth_range"]
depth = example["depth"] / 65535 * (maxv - minv) + minv
with media.set_show_save_dir('/Users/niloofar/Documents/Projects/Video_OF/videoData/'):
    media.show_videos({#"rgb": example["video"], 
                       #"normal": example["normal"],
                       #"forward_flow": flow_to_rgb(forward_flow, white_bg=False),
                       "backward_flow": flow_to_rgb(backward_flow, white_bg=False),
                       "object_coordinates": example["object_coordinates"], 
                    },
                    fps=12,)
                    #columns=4,
                    #title='videoF',)
    
  