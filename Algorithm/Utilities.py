# This script contains common helper functions
from os import listdir
from os.path import isfile, join
import os, sys, argparse
import shutil
import os
import numpy as np
import code
import pylab as pl

### Plotting ###
def plot_image_similarities(plot_title, first_title, nearest, similarities):
  fig = pl.figure()
  fig.clear()
  #pl.title(plot_title)
  
  I = pl.imread(nearest[0].image_path)
  fig.add_subplot(4,3,2)
  #fig.add_subplot(1, len(nearest)+2, 1)
  pl.title(first_title)
  pl.axis('off')
  pl.imshow(I)
  
  for i in range(len(nearest)):
    image = nearest[i]
    I = pl.imread(image.image_path)
    fig.add_subplot(4,3,i+4)
    #fig.add_subplot(1, len(nearest)+2, i+3)
    pl.title(str(similarities[i])[:6])
    pl.axis('off')
    pl.imshow(I)
  pl.show()
  
# Mode: 'day' or 'year'
def plot_cluster_timeline(images, cluster, mode):
  cluster_images = [images[i] for i in cluster]
  #sorted_images =
  #fig = pl.figure()
  #fig.clear()
  
  

### Feature saving/loading ###
def save_features(name, features):
  features = features.toarray()
  n = features.shape[1]
  filename = name + '+' + str(n)
  print "Saving features", filename
  folder = './features/'
  if not os.path.exists(folder):
    os.makedirs(folder)
  np.save(folder + filename, features)
  
def load_features(name, n_codebook):
  features_name = name + '+' + str(n_codebook)
  folder = './features/'
  filename = folder + features_name + '.npy'
  if not os.path.exists(filename):
    print "Features not found for", features_name
    return None
  print "Loading features", features_name
  return np.load(filename)
  
### Codebook saving/loading ### (todo merge with feature saving/loading)

def save_codebook(name, codebook):
  codebook_name = name + '+' + str(len(codebook))
  print "Saving codebook", codebook_name
  folder = './Codebooks/'
  if not os.path.exists(folder):
    os.makedirs(folder)
  codebook = np.array(codebook)
  np.save(folder + codebook_name, codebook)
  
def load_codebook(name, n_codebook):
  codebook_name = name + '+' + str(n_codebook)
  folder = './Codebooks/'
  filename = folder + codebook_name + '.npy'
  if not os.path.exists(filename):
    print "Codebook not found for", codebook_name
    return None
  print "Loading codebook", codebook_name
  return np.load(filename)

### Algorithmic utilities ###
def meters_to_latlong_approx(mx, my):
  return (mx / 111413.93794495543, my / 55631.4933990071)

def latlong_to_meters_approx(lat, long):
  return (lat * 111413.93794495543, long * 55631.4933990071)

### File copying etc ###
def copy_images(input_folder, output_folder, img_paths, md_paths):
  if input_folder == output_folder:
    print "input folder can't be same as output folder!"
    return
  
  if os.path.exists(output_folder):
    shutil.rmtree(output_folder)  # delete folder if it already exists
  if not os.path.exists(output_folder):
    os.makedirs(output_folder)
  #print "Copying {} photos".format(len(img_paths))
  for i in range(len(img_paths)):
    shutil.copy2(input_folder + img_paths[i], output_folder + img_paths[i])
    shutil.copy2(input_folder + md_paths[i], output_folder + md_paths[i])

def get_filename(path): # probably doesn't work on other than Windows
  if '/' not in path:
    return path
  else:
    return path.split('/')[-1]

def get_folder(path):
  return path[:path.rfind('/')+1]
  
def get_clustering_arguments():
  parser = argparse.ArgumentParser(description='This script clusters images of same views')
  parser.add_argument('-f', '--folder_name', help='Image folder name', required=True)
  parser.add_argument('-c', '--codebook_size', help='Visual codebook size', required=True)
  args = parser.parse_args()
  return args

def get_image_paths(folder):
    all_file_names = [f for f in listdir(folder) if isfile(join(folder, f))]
    all_file_names = [f for f in all_file_names if f != 'ratings.txt']
    all_file_names.sort()
    files = [folder + all_file_names[i].split('.')[0] for i in range(0, len(all_file_names), 2)]
    images = [f + '.jpg' for f in files]
    metadata = [f + '.txt' for f in files]
    return (images, metadata)
