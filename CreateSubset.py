from os import listdir
from os.path import isfile, join
import os, sys, argparse
import json
import utilities

def get_folder_arguments():
  parser = argparse.ArgumentParser(description='This script clusters images of same views')
  parser.add_argument('-i', '--input_folder', help='Input image folder name', required=True)
  parser.add_argument('-o', '--output_folder', help='Output image folder name', required=True)
  args = parser.parse_args()
  return (args.input_folder, args.output_folder)

def get_image_names(folder):
    all_file_names = [f for f in listdir(folder) if isfile(join(folder, f))]
    all_file_names = [f for f in all_file_names if f != 'ratings.txt']
    all_file_names.sort()
    files = [all_file_names[i].split('.')[0] for i in range(0, len(all_file_names), 2)]
    images = [f + '.jpg' for f in files]
    metadata = [f + '.txt' for f in files]
    return (images, metadata)

def get_bb_images(folder, image_paths, metadata_paths, latlong, w, h):
  imgs = []
  mds = []
  for i in range(len(image_paths)):
    with open(folder + metadata_paths[i], 'r') as f:
      md = json.load(f)
      gps = [float(md['gps'][0]), float(md['gps'][1])]
      if abs(latlong[0] - gps[0]) < w and abs(latlong[1] - gps[1]) < h:
        imgs.append(image_paths[i])
        mds.append(metadata_paths[i])
  return (imgs, mds)
  
def meters_to_latlong_approx(mx, my):
  return (mx / 111413.93794495543, my / 55631.4933990071)
    
def main():
  (input_folder, output_folder) = get_folder_arguments()
  (image_paths, metadata_paths) = get_image_names('./' + input_folder + '/')
  eduskuntatalo = [60.172538, 24.9333456]
  rautatietori = [60.171267, 24.944136]
  center = eduskuntatalo
  (w, h) = meters_to_latlong_approx(700.0, 700.0)
  print "w: {}, h: {}".format(w, h)
  (subset_images, subset_metadata) = get_bb_images('./' + input_folder + '/', image_paths, metadata_paths, center, w, h)
  utilities.copy_images('./' + input_folder + '/', './' + output_folder + '/', subset_images, subset_metadata)
  
if __name__ == '__main__':
  main()