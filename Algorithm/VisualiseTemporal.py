#!/usr/bin/python
#-*- coding: utf-8 -*-

import pylab as pl
import argparse
import code

from Image import Image
import Utilities

# Comparison functions for different temporal modes
def compare_in_day(i1, i2):
  if i1.date.hour == i2.date.hour:
    return i1.date.minute < i2.date.minute
  return i1.date.hour < i2.date.hour
def compare_in_year(i1, i2):
  if i1.date.month == i2.date.month:
    return i1.date.day < i2.date.day
  return i1.date.month < i2.date.month
def sort_by_time(images, temporal_mode):
  if temporal_mode == 'day':
    images.sort(cmp=compare_in_day)
  if temporal_mode == 'year':
    images.sort(cmp=compare_in_year)
  
# Visualisation of temporal change of cluster of images
def visualise_temporal(images, temporal_mode):
  image_bins = [[] for i in range(12)]
  for image in images:
    if temporal_mode == 'day':
      index = (image.date.hour - 1) / 2
    if temporal_mode == 'year':
      index = (image.date.month - 1) / 2
    image_bins[index].append(image)
  
  # Plot images
  fig = pl.figure()
  fig.clear()
  pl.axis('off')
  for i in range(len(image_bins)):
    bin = image_bins[i]
    fig.add_subplot(1,12,i+1)
    pl.axis('off')
    if temporal_mode == 'day':
      pl.title(i*2 + 1)
    if temporal_mode == 'year':
      pl.title(i+1)
    if (len(bin) == 0):
      continue
    I = pl.imread(bin[0].image_path)
    pl.imshow(I)
  pl.show()
  
def read_args():
  parser = argparse.ArgumentParser(description='This script clusters images of same views')
  parser.add_argument('-f', '--folder_name', help='Image folder name', required=True)
  parser.add_argument('-t', '--temporal_mode', help='Temporal mode ("day", "year")', required=True)
  return parser.parse_args()
  
def main():
  args = read_args()
  base_folder = './Clusters/' + args.folder_name + '/'
  (image_paths, metadata_paths) = Utilities.get_image_paths(base_folder)
  images = []
  for i in range(len(image_paths)):
    images.append(Image(image_paths[i], metadata_paths[i]))
  visualise_temporal(images, args.temporal_mode)
  
if __name__ == '__main__':
  main()