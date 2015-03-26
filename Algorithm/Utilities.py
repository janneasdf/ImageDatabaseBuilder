# This script contains common helper functions
from os import listdir
from os.path import isfile, join
import os, sys, argparse
import shutil
import os
import numpy as np
import pylab
import code

### Plotting ###
def plot_image_similarities(first_title, nearest, similarities):
  fig = pylab.figure()
  fig.clear()
  
  I = pylab.imread(nearest[0].image_path)
  fig.add_subplot(6,5,3)
  pylab.title(first_title)
  pylab.axis('off')
  pylab.imshow(I)
  
  for i in range(len(nearest)):
    image = nearest[i]
    I = pylab.imread(image.image_path)
    fig.add_subplot(6,5,i+6)
    pylab.title(str(similarities[i])[:6])
    pylab.axis('off')
    pylab.imshow(I)
  pylab.show()

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
  
def get_folder_argument():
  parser = argparse.ArgumentParser(description='This script clusters images of same views')
  parser.add_argument('-f', '--folder_name', help='Image folder name', required=True)
  args = parser.parse_args()
  return args.folder_name

def get_image_paths(folder):
    all_file_names = [f for f in listdir(folder) if isfile(join(folder, f))]
    all_file_names = [f for f in all_file_names if f != 'ratings.txt']
    all_file_names.sort()
    files = [folder + all_file_names[i].split('.')[0] for i in range(0, len(all_file_names), 2)]
    images = [f + '.jpg' for f in files]
    metadata = [f + '.txt' for f in files]
    return (images, metadata)
