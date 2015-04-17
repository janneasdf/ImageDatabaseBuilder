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
  
def plot_time_distributions(images, temporal_mode):
  if temporal_mode == 'day':
    #freq_bins = [0] * 24
    hours = []
    for image in images:
      #freq_bins[image.date.hour - 1] += 1
      hours.append(image.date.hour)
    #pl.hist(freq_bins)
    pl.hist(hours, bins=24)
    pl.title(u"Kuvien määrä kuvanottamisajan tunnin suhteen")
    pl.xlabel(u"Tunti")
    pl.ylabel(u"Kuvien määrä")
    pl.xlim(1, 24)
    pl.show()
  if temporal_mode == 'year':
    #freq_bins = [0] * 12
    months = []
    for image in images:
      #freq_bins[image.date.month - 1] += 1
      months.append(image.date.month)
    #pl.hist(freq_bins)
    pl.hist(months, bins=12)
    pl.title(u"Kuvien määrä kuvanottamisajan kuukauden suhteen")
    pl.xlabel(u"Kuukausi")
    pl.ylabel(u"Kuvien määrä")
    pl.xlim(1, 12)
    pl.show()
  
def read_args():
  parser = argparse.ArgumentParser(description='This script clusters images of same views')
  parser.add_argument('-f', '--folder_name', help='Image folder name', required=True)
  parser.add_argument('-t', '--temporal_mode', help='Temporal mode ("day", "year")', required=True)
  return parser.parse_args()
  
def main():
  args = read_args()
  base_folder = '../' + args.folder_name + '/'
  (image_paths, metadata_paths) = Utilities.get_image_paths(base_folder)
  images = []
  for i in range(len(image_paths)):
    images.append(Image(image_paths[i], metadata_paths[i]))
  plot_time_distributions(images, args.temporal_mode)
  
if __name__ == '__main__':
  main()