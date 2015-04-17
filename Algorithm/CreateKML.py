#!/usr/bin/python
#-*- coding: utf-8 -*-

import pylab as pl
import argparse
import code

from Image import Image
import Utilities

def create_KML(images, folder_name):
  gpses = [image.gps for image in images]
  output = ''
  output = output + '<?xml version="1.0" encoding="UTF-8"?>'
  output = output + '<kml xmlns="http://www.opengis.net/kml/2.2">'
  output = output + '<Document>'
  for gps in gpses:
    gps_string = str(gps[1]) + ',' + str(gps[0]) #str(gps[0]) + ',' + str(gps[1])
    output = output + '<Placemark><Point><coordinates>' + gps_string + '</coordinates></Point></Placemark>'
  output = output + '</Document>'
  output = output + '</kml>'
  with open('KML/' + folder_name + '.kml', 'w') as f:
    f.write(output)

def read_args():
  parser = argparse.ArgumentParser(description='This script creates KML file from image gpses')
  parser.add_argument('-f', '--folder_name', help='Image folder name', required=True)
  return parser.parse_args()
  
def main():
  args = read_args()
  base_folder = '../' + args.folder_name + '/'
  (image_paths, metadata_paths) = Utilities.get_image_paths(base_folder)
  images = []
  for i in range(len(image_paths)):
    images.append(Image(image_paths[i], metadata_paths[i]))
  create_KML(images, args.folder_name)
  
if __name__ == '__main__':
  main()