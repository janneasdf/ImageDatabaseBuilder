#!/usr/bin/python
#-*- coding: utf-8 -*-
import Utilities
import numpy as np
import code
import pylab as pl
from sklearn.metrics.pairwise import cosine_similarity

def plot_distribution(vals, title, xlabel, ylabel, bins):
  pl.hist(vals, bins=bins)
  pl.title(title)
  pl.xlabel(xlabel)
  pl.ylabel(ylabel)
  pl.show()

def save_clusters(images, labels, folder):
  clusters = {}
  for l in np.unique(labels):
    clusters[l] = []
  for i in range(len(labels)):
    clusters[labels[i]].append(i)
  for c in clusters:
    for index in clusters[c]:
      image = images[index]
  # Save clusters for easy viewing
  base_output_folder = './Clusters/' + folder + '/'
  print "Saving clusters to {}".format(base_output_folder)
  for c in clusters:
    input_folder = Utilities.get_folder(images[clusters[c][0]].image_path)
    output_folder = base_output_folder + '{}/'.format(c)
    img_paths = []
    md_paths = []
    for index in clusters[c]:
      image = images[index]
      filename = Utilities.get_filename(image.image_path).split('.')[0] # get rid of folder and extension
      img_paths.append(filename + '.jpg')
      md_paths.append(filename + '.txt')
      #print filename
    #print "Copying cluster {} files from {} to {}".format(c, input_folder, output_folder)
    Utilities.copy_images(input_folder, output_folder, img_paths, md_paths)

# Plots nearest neighbors visually and then by tags and then together
def plot_similarities(image_index, images, n_nearest, visual_tfidf, tags_tfidf, ext_tags_tfidf, gpses, S):
  def plot_sims(plot_title, first_title, similarities, reverse):
    image_similarities = zip(similarities[image_index], range(similarities.shape[1]))
    nearest_pairs = sorted(image_similarities, key=lambda p: p[0], reverse=reverse)
    nearest_pairs = nearest_pairs[:n_nearest]
    nearest_indices = [pair[1] for pair in nearest_pairs]
    nearest_images = [images[i] for i in [pair[1] for pair in nearest_pairs]]
    nearest_sims = [pair[0] for pair in nearest_pairs]
    Utilities.plot_image_similarities(plot_title, first_title, nearest_images, nearest_sims)
  # Plot by similarity of visual features
  similarities = cosine_similarity(visual_tfidf)
  #plot_sims('Visuaalisten piirteiden samanlaisuus', '', similarities, True)  
  # Plot by similarity of tags
  similarities = cosine_similarity(tags_tfidf)
  #plot_sims('Avainsanojen samanlaisuus', ' '.join(images[image_index].tags), similarities, True)  
  # Plot by similarity of extended_tags
  similarities = cosine_similarity(ext_tags_tfidf)
  #plot_sims('Laajennettujen avainsanojen samanlaisuus', ' '.join(images[image_index].extended_tags), similarities, True)  
  # Plot by distance
  similarities = S
  plot_sims(u'Kuvien etäisyys d(x, y)', u'', similarities, False)

def plot_gps_distribution(images):
  gpses = [image.gps for image in images]
  lat_h = [gps[0] for gps in gpses]
  lon_h = [gps[1] for gps in gpses]
  lat150m = []
  lon150m = []
  center = [60.172538, 24.9333456]
  w, h = Utilities.meters_to_latlong_approx(150.0, 150.0)
  for image in images:
    gps = image.gps
    if abs(center[0] - gps[0]) < w and abs(center[1] - gps[1]) < h:
      lat150m.append(gps[0])
      lon150m.append(gps[1])
  
  import pylab as pl
  fig = pl.figure()
  ax = fig.add_subplot(111)
  code.interact(local=locals())
  ax.scatter(lat_h, lon_h, c='b', label=u'Kaikki kuvat')
  ax.scatter(lat150m, lon150m, c='r', label=u'Kuvat noin 150m säteellä Eduskuntatalosta')
  
  #pl.xlim(min(lat), max(lat))
  #pl.ylim(min(lon), max(lon))
  #pl.xlim(59.977005 - 0.05, 60.2891 + 0.05)
  #pl.ylim(24.80288 - 0.05, 25.3125 + 0.05)
  pl.xlabel('Leveyspiiri')
  pl.ylabel('Pituuspiiri')
  pl.show()
  #print min(lat), max(lat)
  #print min(lon), max(lon)

def find_time_correlated_tags(images):
  vocabulary = []
  for image in images:
    for tag in image.tags:
      if tag not in vocabulary:
        vocabulary.append(tag)
  hourly_hists = [[0] * len(vocabulary)] * 24
  monthly_hists = [[0] * len(vocabulary)] * 12
  for image in images:
    hour = image.date.hour
    month = image.date.month - 1
    for i in range(len(vocabulary)):
      if vocabulary[i] in image.tags:
        monthly_hists[month][i] += 1
        hourly_hists[hour][i] += 1
  from sklearn.feature_extraction.text import TfidfTransformer
  hourly_hists = np.array(hourly_hists).transpose()
  monthly_hists = np.array(monthly_hists).transpose()
  hourly_tfidf = TfidfTransformer().fit_transform(hourly_hists).toarray()
  monthly_tfidf = TfidfTransformer().fit_transform(monthly_hists).toarray()
  
  monthly_favs = []
  for i in range(12):
    monthly_favs.append(vocabulary[np.argmax(monthly_tfidf[i])])
  hourly_favs = []
  for i in range(24):
    hourly_favs.append(vocabulary[np.argmax(hourly_tfidf[i])])
  
  code.interact(local=locals())
  
  '''hourly_tag_hists = [[0] * len(vocabulary)] * 24
  monthly_tag_hists = [[0] * len(vocabulary)] * 12
  for image in images:
    hour = image.date.hour
    month = image.date.month - 1  # 0-indexing
    for tag, i in zip(vocabulary, range(len(vocabulary))):
      if tag in image.tags:
        monthly_tag_hists[month][i] += 1
        hourly_tag_hists[hour][i] += 1
  from sklearn.feature_extraction.text import TfidfTransformer
  hourly_tag_hists = np.array(hourly_tag_hists).transpose()
  monthly_tag_hists = np.array(monthly_tag_hists).transpose()
  hourly_tfidf = TfidfTransformer().fit_transform(hourly_tag_hists)
  monthly_tfidf = TfidfTransformer().fit_transform(monthly_tag_hists)
  for i in range(24):
    hourly_max = max(hourly_tfidf[i])
    import code
    code.interact(local=locals())
    for j in range(hourly_tfidf.shape[0]):
      if hourly_tfidf[i, j] == hourly_max:
        print "max:", vocabulary[j]'''
