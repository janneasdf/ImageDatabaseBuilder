#!/usr/bin/python
#-*- coding: utf-8 -*-

import argparse, json
import math
import string
from os import listdir
from os.path import isfile, join
import pylab as pl
from math import sqrt
import utilities

class ImageInfo:
  def __init__(self, filename, rating):
    self.score = None
    self.filename = filename
    self.rating = rating
    if self.rating != None:
      self.rating = int(rating)
    if self.rating == 0:
      self.rating = -1
    with open(filename + '.txt', 'r') as f:
      md = json.load(f)
      self.metadata = md
      self.tags = md['tags']
      self.tags = [tag.lower() for tag in self.tags if len(tag) > 0]
      #self.load_extended_tags()
      
  def load_extended_tags(self):
    self.tags = [tag.lower() for tag in self.metadata['tags']]
    # Extended tags = tags + title + description
    def add_tags_from_text(words, text):
      for word in text.split(' '):
        word = word.replace(' ', '').lower()
        if len(word) == 0:
          continue
        if word not in words:
          words.append(word)
      return words
    self.tags = add_tags_from_text(self.tags, self.metadata['title'])
    self.tags = add_tags_from_text(self.tags, self.metadata['description'])

def get_images_in_folder(folder):
    all_file_names = [f for f in listdir(folder) if isfile(join(folder, f))]
    all_file_names = [f for f in all_file_names if f != 'ratings.txt']
    all_file_names.sort()
    files = [folder + all_file_names[i].split('.')[0] for i in range(0, len(all_file_names), 2)]
    return files

def create_hists_and_vocab(images):
  bad_hist = {}
  good_hist = {}
  all_tags = []
  for img in images:
    for tag in img.tags:
      if tag not in all_tags:
        all_tags.append(tag)
      bad_hist[tag] = 0
      good_hist[tag] = 0
  for img in images:
    for tag in img.tags:
      if img.rating == 1:
        good_hist[tag] += 1
      elif img.rating == -1:
        bad_hist[tag] += 1
  return (bad_hist, good_hist, all_tags)

def normalize_data(data, norm):
  if norm == '0':
    pass
  elif norm == 'L1':
    data_sum = sum(data)
    data = [float(d) / data_sum for d in data]
  elif norm == 'L2':
    sq_sum = 0.0
    for d in data:
      sq_sum += d*d
    sq_sum = sqrt(sq_sum)
    data = [d / sq_sum for d in data]
  else:
    raise Exception('Norm not recognized')
  return data
  
def main(training_folder, classify_folder, norm, save_subsets):
  # Read the manually created ratings
  with open('./' + training_folder + '/ratings.txt', 'r') as f:
    ratings = json.load(f)
    
  # Read rated images
  rated_images = []
  for filename in ratings:
    if ratings[filename] == None:
      continue
    img = ImageInfo(filename, ratings[filename])
    rated_images.append(img)
  
  # Create histograms and all_tags (vocabulary)
  (bad_hist, good_hist, all_tags) = create_hists_and_vocab(rated_images)
  
  # HISTOGRAM PLOTTING
  all_tags.sort(key=lambda tag: good_hist[tag] + bad_hist[tag], reverse=True)
  all_training_tags = all_tags
  all_tags = all_tags[:20]
  show_tags = True
  fig = pl.figure()
  good_data = [good_hist[tag] for tag in all_tags]
  bad_data = [-bad_hist[tag] for tag in all_tags]
  
  # Normalization
  good_data = normalize_data(good_data, norm)
  if norm == 'L1':
    bad_data = [-d for d in bad_data]
  bad_data = normalize_data(bad_data, norm)
  if norm == 'L1':
    bad_data = [-d for d in bad_data]
  
  if norm == '0':
    pl.title('Avainsanojen frekvenssi')
  else:
    pl.title('Avainsanojen {}-normalisoitu frekvenssi'.format(norm))
  ax = pl.subplot(111)
  ax.bar(range(len(all_tags)), good_data, width=1, color='b')
  ax.bar(range(len(all_tags)), bad_data, width=1, color='r')
  pl.ylabel('frekvenssi')
  pl.xlabel('avainsanan indeksi')
  if show_tags:
    pl.xticks(range(len(all_tags)), all_tags)
    pl.xticks(rotation=90)
    pl.xlabel('avainsana')
    pl.gcf().subplots_adjust(bottom=0.45)
  
  ax.legend((u'Hyödylliset kuvat', u'Hyödyttömät kuvat'), 'lower right')
    
  #pl.show()
  
  # Divide images to training and test images
  import numpy as np
  import random
  def occurrance_matrix(images, vocabulary):
    X = np.zeros((len(images), len(vocabulary)))
    for i in range(len(images)):
      for j in range(len(vocabulary)):
        if vocabulary[j] in images[i].tags:
          X[(i, j)] = 1
    return X
  def leave_one_out(training_images, test_image, vocab):
    X_training = occurrance_matrix(training_images, vocab)
    Y_training = np.array([img.rating for img in training_images])
    X_testing = occurrance_matrix([test_image], vocab)
    from sklearn.naive_bayes import BernoulliNB
    classifier = BernoulliNB()
    classifier.fit(X_training, Y_training)
    classifier.classes_ = np.array([-1,1])
    estimates = classifier.predict(X_testing)
    if estimates[0] == test_image.rating:
      return 1.0
    else:
      return 0.0
    
  repetitions_per_n = 100
  clf_rates = []
  training_ns = range(10, len(rated_images), 2)
  for n_training in training_ns:
    correct = 0.0
    print "training: {}".format(n_training)
    for repetition in range(repetitions_per_n):
      clf_images = [image for image in rated_images]
      random.shuffle(clf_images)
      correct += leave_one_out(clf_images[:n_training], clf_images[n_training], all_training_tags)
    clf_rates.append(correct / repetitions_per_n)
  print clf_rates
  good_imgs = 0
  for image in rated_images:
    if image.rating == 1:
      good_imgs += 1
  a_priori = float(good_imgs) / len(rated_images)
  pl.figure()
  pl.clf()
  pl.plot(training_ns, clf_rates, 'b-', training_ns, [0.5] * len(training_ns), 'r-', training_ns, [a_priori] * len(training_ns), 'g-')
  pl.legend(['Naiivi Bayes-luokitin', 'Satunnaisluokitin', 'A priori -luokitin'], loc='lower right')
  pl.xlabel(u'Opetuskuvien määrä')
  pl.ylabel(u'Luokittelun onnistumistodennäköisyys')
  pl.ylim(0.0, 1.0)
  pl.xlim(training_ns[0], training_ns[-1])
  pl.show()
  
  rand_subset = range(len(rated_images))
  random.shuffle(rand_subset)
  #n_training = len(rated_images) / 4 #len(rated_images) - 40
  print "Training data size: {}, Testing data size: {}".format(n_training, len(rated_images)-n_training)
  training_images = [rated_images[i] for i in rand_subset[:n_training]]
  test_images = [rated_images[i] for i in rand_subset[n_training:]]
  X_training = occurrance_matrix(training_images, all_training_tags)
  Y_training = np.array([img.rating for img in training_images])
  X_testing = occurrance_matrix(test_images, all_training_tags)
  Y_testing = np.array([img.rating for img in test_images])
  # Train the Bernoulli Naive Bayes
  from sklearn.naive_bayes import BernoulliNB
  classifier = BernoulliNB()
  classifier.fit(X_training, Y_training)
  
  mean_accuracy = classifier.score(X_testing, Y_testing)
  print "Mean accuracy:", mean_accuracy
  
  estimates = classifier.predict(X_testing)
  good_to_bad_errors = 0
  bad_to_good_errors = 0
  for i in range(estimates.shape[0]):
    if estimates[i] == -1 and Y_testing[i] == 1:
      good_to_bad_errors += 1
    if estimates[i] == 1 and Y_testing[i] == -1:
      bad_to_good_errors += 1
  print "Bad images classified as good:", float(bad_to_good_errors) / estimates.shape[0]
  print "Good images classified as bad:", float(good_to_bad_errors) / estimates.shape[0]
  
  # todo kokeile tf-idf luokitinta tms
  
  # Classify ALL images, using ALL rated images for training (todo, use ratings of subset or whole set?)
  if classify_folder == None:
    return
  all_image_files = get_images_in_folder('./' + classify_folder + '/')
  all_images = [] # all images in the classify folder except ones also in subset
  for img in all_image_files:
    in_subset = False
    for rated_img in rated_images:
      if img == rated_img.filename:
        print img
        in_subset = True
        break
    if not in_subset:
      all_images.append(ImageInfo(img, None))
  (_, _, total_vocab) = create_hists_and_vocab(all_images)
  X_training = occurrance_matrix(rated_images, total_vocab)
  Y_training = np.array([img.rating for img in rated_images])
  X_classify = occurrance_matrix(all_images, total_vocab)
  classifier = BernoulliNB()
  classifier.fit(X_training, Y_training)
  all_estimates = classifier.predict(X_classify)
  if save_subsets:
    copy_from_folder = utilities.get_folder(all_images[0].filename)
    copy_to_base_folder = './' + args.training_images + '+' + args.classify_images + '/'
    print copy_from_folder, copy_to_base_folder
    good_img_paths = []
    good_md_paths = []
    bad_img_paths = []
    bad_md_paths = []
    for i in range(len(all_estimates)):
      filename = utilities.get_filename(all_images[i].filename)
      if all_estimates[i] == 1: # image classified as "good"
        good_img_paths.append(filename + '.jpg')
        good_md_paths.append(filename + '.txt')
      else: # image classified as "bad"
        bad_img_paths.append(filename + '.jpg')
        bad_md_paths.append(filename + '.txt')
    utilities.copy_images(copy_from_folder, copy_to_base_folder + 'good_images/', good_img_paths, good_md_paths)
    utilities.copy_images(copy_from_folder, copy_to_base_folder + 'bad_images/', bad_img_paths, bad_md_paths)
  
  
if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='This script analyses tag usefulness')
  parser.add_argument('-f1', '--training_images', help='Folder name of training (rated) images', required=True)
  parser.add_argument('-f', '--classify_images', help='Folder name of images to classify', required=False, default=None)
  parser.add_argument('-n', '--norm', help='Normalization method (0, L1, L2)', required=True)
  parser.add_argument('-s', '--save_subsets', action='store_true', help='Save good and bad images?', default=False)
  args = parser.parse_args()
  
  main(args.training_images, args.classify_images, args.norm, args.save_subsets)