#!/usr/bin/python
#-*- coding: utf-8 -*-

import numpy as np
import math
import random
import code
import cv2
import json
import sklearn
from sklearn.metrics.pairwise import cosine_similarity

import Utilities
import ClusteringHelpers
from Image import Image

import os
import sys
  
def create_visual_codebook(images, n_codebook, n_maxfeatures, n_maxdescriptors):
  descriptors = []
  surf = cv2.SURF(400)
  failed_images = 0
  print "Extracting descriptors from images"
  try:
    for image, i in zip(images, range(len(images))):
        img = cv2.imread(image.image_path)
        kp, des = surf.detectAndCompute(img, None)
        if des == None:
          failed_images += 1
          continue
        des = des[:n_maxfeatures]
        descriptors.extend(des)
        #image.des = des
  except KeyboardInterrupt as e:
    raise e
  except:
    print "Failed at image {}/{}".format(i+1, len(images))
  
  print "Found {} descriptors".format(len(descriptors))
  
  if n_maxdescriptors != None:
    random.shuffle(descriptors)
    descriptors = descriptors[:n_maxdescriptors]
  
  print "Clustering {} descriptors into codebook of size {}".format(len(descriptors), n_codebook)
  #if n_codebook > len(descriptors) / 5:
  #  n_codebook = len(descriptors) / 5
  #  print "Changing n_codebook to {} ({} descriptors)".format(n_codebook, len(descriptors))
  from sklearn.cluster import MiniBatchKMeans
  mbk = MiniBatchKMeans(init='k-means++', n_clusters=n_codebook, n_init=3, max_iter=50, max_no_improvement=3, verbose=0, compute_labels=False, batch_size=50) # batch size?
  mbk.fit(descriptors)
  codebook = mbk.cluster_centers_
  '''try:
    codebook = np.array(descriptors[:n_codebook]) # testing random clusters
  except:
    code.interact(local=locals())'''
  return codebook

# Generates tfidf-weighted BoW vectors (histograms) for images
def generate_histograms(images, codebook):
  import scipy
  import scipy.cluster.vq
  
  def generate_histogram(codebook, features):  # from vocpy library
        [N, d] = codebook.shape
        if features.size <= 1:
            return np.zeros((N, 0))

        [hits, d] = scipy.cluster.vq.vq(features, codebook)
        [y, x] = np.histogram(hits, bins=range(0, N + 1))
        return y
  
  print "Generating feature histograms"
  n_codebook = len(codebook)
  visual_features = []
  surf = cv2.SURF(400)
  for image in images:
    img = cv2.imread(image.image_path)
    _, des = surf.detectAndCompute(img, None)
    if des != None:
      visual_hist = generate_histogram(codebook, des)#image.des)
    else:
      visual_hist = [math.sqrt(1.0 / n_codebook)] * n_codebook # flat histogram
    visual_features.append(visual_hist)
  from sklearn.feature_extraction.text import TfidfTransformer
  visual_tfidf = TfidfTransformer(norm=u'l2', use_idf=True, smooth_idf=True, sublinear_tf=False)
  visual_features = visual_tfidf.fit_transform(visual_features)
  return visual_features
  
def compute_tag_tfidf(images, use_extended_tags):
  # Create tag vocabulary
  print "Creating tag vocabulary"
  vocabulary = []
  tags = []
  tag_type = ['tags', 'extended_tags'][use_extended_tags]
  for image in images:
    for tag in getattr(image, tag_type):
      if tag not in vocabulary:
        vocabulary.append(tag)
    tags.append(' '.join(image.tags))
  
  # Find time-specific tags and remove them from vocabulary/tags
  #time_tags = find_time_correlated_tags(images)
  
  # Create TF-IDF features from each image's tags
  print "Computing tf-idf tag features for images"
  from sklearn.feature_extraction.text import TfidfVectorizer
  tfidf_vectorizer = TfidfVectorizer()
  tags_tfidf = tfidf_vectorizer.fit_transform(tags)
  #print "tfidf matrix shape:", tags_tfidf.shape
  #print "vocab size: {}, tfidf size: {}".format(len(vocabulary), tags_tfidf.shape[1])
  return tags_tfidf
    
# Combine visual and tag features
def combine_features(visual_tfidf, tag_tfidf):
  import scipy.sparse
  from sklearn.preprocessing import normalize
  print visual_tfidf.shape
  print tag_tfidf.shape
  features = scipy.sparse.hstack((visual_tfidf, tag_tfidf))
  features = normalize(features, axis=1, norm='l2')
  return features

# This function searches for clusters
def find_view_clusters(features):
  n = features.shape[0]
  from sklearn.cluster import MeanShift, DBSCAN, KMeans, SpectralClustering
  ms = MeanShift(bandwidth=0.95)
  labels = ms.fit_predict(features)
  print "MeanShift Labels, unique, n:", labels, len(np.unique(labels)), n
  db = DBSCAN(eps=1.05)
  #labels = db.fit_predict(features)
  print "DBSCAN Labels, unique, n:", labels, len(np.unique(labels)), n
  #sc = SpectralClustering()
  #labels = sc.fit_predict(features)
  #print "Spectral Labels:", labels
  #km = KMeans(n_clusters=n/5)
  #labels = km.fit_predict(features)
  #print "KMeans Labels:", labels
  
  return labels

def geo_dist_approx(latlong1, latlong2):
  delta = (latlong1[0] - latlong2[0], latlong1[1] - latlong2[1])
  delta = (abs(delta[0]), abs(delta[1]))
  mx, my = Utilities.latlong_to_meters_approx(delta[0], delta[1])
  dist = math.sqrt(mx*mx + my*my)
  #print "Dist:",dist
  return dist

def common_tags(image1, image2):
  common = []
  for tag in image1.tags:
    if tag in image2.tags:
      common.append(tag)
  return common
  
def image_distance((image1, vis1, tag1), (image2, vis2, tag2)):
  '''d = 1.0 / (1.0 + vis1.dot(vis2))
  if image1.metadata['owner'] == image2.metadata['owner']:
    if abs((image1.date - image2.date).total_seconds()) < 15 * 60:
      d *= 0.5
  if geo_dist_approx(image1.gps, image2.gps) > 200.0:
    d *= 2.0
  d = d / (1.0 + tag1.dot(tag2))'''
  # Visual similarity
  s_vis = vis1.dot(vis2)
  # Time taken similarity
  s_time = 0.0
  if image1.metadata['owner'] == image2.metadata['owner']:
    if abs((image1.date - image2.date).total_seconds()) < 15 * 60:
      s_time = 1.0
  s_geo = 0.0
  # Geographic similarity
  if geo_dist_approx(image1.gps, image2.gps) < 1500.0:
    s_geo = 1.0
  # Tag similarity
  s_tag = tag1.dot(tag2)
  
  # Coefficients for different types of similarities
  c_vis = 0.4
  c_tag = 0.05
  c_time = 0.05
  c_geo = 0.5
  
  # Distance
  d = 1.0 - c_vis * s_vis - c_tag * s_tag - c_time * s_time - c_geo * s_geo
  #d = 1.0 - s_vis
  #if s_geo == 0.0:
  #  d = 100.0
  
  # Clamp distance to [0, 1]
  d = max(0.0, d)
  d = min(1.0, d)
  
  return d

# Not sure if this can be used
def compute_similarity(visual_tfidf, tags_tfidf, gpses, images):
  n = len(images)
  S = np.zeros((n, n))
  
  # make distance matrix
  for i in range(n):
    for j in range(n):
      if i == j:
        S[i, j] = 0.0 # same picture is identical
        continue
      d = image_distance((images[i], visual_tfidf[i], tags_tfidf[i]), (images[j], visual_tfidf[j], tags_tfidf[j]))
      S[i, j] = d
      if '15791343418_b8c738bf32_z' in images[i].image_path and '15991895575_3705685e6c_z' in images[j].image_path:
        print "ed & tuomiokirkko:", d
      if '15791343418_b8c738bf32_z' in images[i].image_path and '16058603384_4b3d947b2e_z' in images[j].image_path:
        print "ed1 & ed2:", d
  
  return S

def cluster_similarity(S, epsilon, min_pts):
  from sklearn.cluster import MeanShift, DBSCAN, KMeans, SpectralClustering
  
  # distance distribution
  n = S.shape[0]
  dists = []
  for i in range(n):
    for j in range(i+1, n):
      dists.append(S[i,j])
  ClusteringHelpers.plot_distribution(dists, u"Kuvien välisten etäisyyksien jakautuminen", u"Etäisyys", u"Frekvenssi", 100)
  print "max d: {}, min d: {}".format(max(dists), min(dists))
  
  print "clustering images"
  #max_cluster_d = 0.85
  #max_cluster_d = 0.7
  print "cluster params: eps={}, min_pts={}".format(epsilon, min_pts)
  dbs = DBSCAN(eps=epsilon, metric='precomputed', min_samples=min_pts)
  labels = dbs.fit_predict(S)
  #print "Labels:", labels
  print "Unique: {} of {}".format(len(np.unique(labels)), S.shape[0])
  
  
  return labels
  
def cluster_by_tags_and_gps(images, folder, n_codebook, epsilon, min_pts):
  #images = images[:1000]
  # Shuffle images
  n_images = len(images)
  n_codebook = n_codebook
  print "n_images:", n_images, "n_codebook:", n_codebook
  #random.shuffle(images)
  
  # Check if visual features have already been computed
  visual_tfidf = Utilities.load_features(folder, n_codebook)
  if visual_tfidf == None:
    # Create visual codebook
    codebook = Utilities.load_codebook(folder, n_codebook)
    if codebook == None:
      n_maxdescriptors = 50000
      n_maxfeatures = 1000
      n_images = len(images)
      print "Creating visual codebook"
      codebook = create_visual_codebook(images, n_codebook, n_maxfeatures, n_maxdescriptors)
      Utilities.save_codebook(folder, codebook)
    visual_tfidf = generate_histograms(images, codebook)
    Utilities.save_features(folder, visual_tfidf)
  
  tags_tfidf = compute_tag_tfidf(images, False)
  ext_tags_tfidf = compute_tag_tfidf(images, True)
  
  # Combine visual and tag features
  print "Combining features"
  features = combine_features(visual_tfidf, tags_tfidf) # todo tsekkaa onko tags_tfidf ees toimiva
  
  tags_tfidf = tags_tfidf.toarray()
  gpses = np.array([image.gps for image in images])
  similarity_matrix = compute_similarity(visual_tfidf, tags_tfidf, gpses, images)
  
  print "Clustering views"
  #clusters = find_view_clusters(features.toarray())
  #clusters = find_view_clusters(visual_tfidf) # no tags
  clusters = cluster_similarity(similarity_matrix, epsilon, min_pts)
  
  # Disqualify clusters where only 1 photographer has photos
  '''cluster_owners = {}
  for cluster in clusters:
    cluster_owners[cluster] = []
  for i in range(len(images)):
    cluster = clusters[i]
    owner = images[i].metadata['owner']
    if owner not in cluster_owners[cluster]:
      cluster_owners[cluster].append(owner)
  disq_clusters = []
  for cluster in cluster_owners:
    if len(cluster_owners[cluster]) < 2:
      disq_clusters.append(cluster)
  print "disq clusters:", disq_clusters
  for i in range(len(clusters)):
    if clusters[i] != -1:
      if clusters[i] in disq_clusters:
        #clusters[i] = -clusters[i]-2  # 0 -> -2, etc
        clusters[i] += 10000 # make them easy to distinguish'''
  
  ClusteringHelpers.save_clusters(images, clusters, folder + '+' + str(n_codebook) + '+' + str(epsilon)[:4] + '+' + str(min_pts))
  
  # Plot similarities and images (really ugly code)
  eduskuntatalo = None
  tuomiokirkko = None
  n_nearest = 9 # how many nearest hits to show
  for img, i in zip(images, range(len(images))):
    #if '15791343418_b8c738bf32_z' in img.image_path:  # image of eduskuntatalo
    if '14607263940_3cc8305502_z' in img.image_path:
      eduskuntatalo = i
    elif '15991895575_3705685e6c_z' in img.image_path:  # helsingin tuomiokirkko
      tuomiokirkko = i
  if eduskuntatalo != None:
    ClusteringHelpers.plot_similarities(eduskuntatalo, images, n_nearest, visual_tfidf, tags_tfidf, ext_tags_tfidf, gpses, similarity_matrix)
  if tuomiokirkko != None:
    pass #ClusteringHelpers.plot_similarities(tuomiokirkko, images, n_nearest, visual_tfidf, tags_tfidf, ext_tags_tfidf, gpses, similarity_matrix)
  
  #ClusteringHelpers.find_time_correlated_tags(images)
  
  #code.interact(local=locals())

def main():
  args = Utilities.get_clustering_arguments()
  base_folder = '../' + args.folder_name + '/'
  n_codebook = int(args.codebook_size)
  
  # Read metadata from files
  (image_paths, metadata_paths) = Utilities.get_image_paths(base_folder)
  images = []
  for i in range(len(image_paths)):
    images.append(Image(image_paths[i], metadata_paths[i]))
  
  # Plot distribution (GPS)
  #ClusteringHelpers.plot_gps_distribution(images)
  
  # Start the main algorithm
  cluster_by_tags_and_gps(images, args.folder_name, n_codebook, args.epsilon, args.min_pts)
  

if __name__ == '__main__':
  main()