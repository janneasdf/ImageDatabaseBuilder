import numpy as np
import math
import random
import code
import cv2
import json
import sklearn
from sklearn.metrics.pairwise import cosine_similarity

import Utilities
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
  mbk = MiniBatchKMeans(init='k-means++', n_clusters=n_codebook, n_init=3, max_iter=50, max_no_improvement=3, verbose=0, compute_labels=False) # batch size?
  mbk.fit(descriptors)
  codebook = mbk.cluster_centers_
  
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
  
def find_time_correlated_tags(images):
  # todo: this should work so that it is done before vocabulary thing
  hourly_tag_hists = [[0] * len(vocabulary)] * 24
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
        print "max:", vocabulary[j]
  
def geo_dist_approx(latlong1, latlong2):
  delta = (latlong1[0] - latlong2[0], latlong1[1] - latlong2[1])
  delta = (abs(delta[0]), abs(delta[1]))
  mx, my = Utilities.latlong_to_meters_approx(delta[0], delta[1])
  dist = math.sqrt(mx*mx, my*my)
  print "Dist:",dist
  return dist
  
def similarity(vis1, tag1, gps1, vis2, tag2, gps2):
  s_vis = vis1.dot(vis2)
  s_tag = tag1.dot(tag2)
  geo_dist = geo_distance(gps1, gps2)
  
# Not sure if this can be used
def compute_similarity(visual_tfidf, tags_tfidf, gpses, images):
  n = len(images)
  S = np.zeros((n, n))
  print "Computing similarity"
  for i in range(n):
    for j in range(i, n):
      s_vis = visual_tfidf[i].dot(visual_tfidf[j])  # todo euclidean or not?
      s_tag = tags_tfidf[i].dot(tags_tfidf[j])
      s_gps = 0.0
      sim = s_vis + s_tag + s_gps
      S[i, j] = sim
      S[j, i] = sim
    
  return S

def cluster_similarity(S):
  pass
  
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
  features = features.toarray()
  n = features.shape[0]
  from sklearn.cluster import MeanShift, DBSCAN, KMeans, SpectralClustering
  ms = MeanShift()
  labels = ms.fit_predict(features)
  print "MeanShift Labels:", labels[:10]
  db = DBSCAN()
  labels = db.fit_predict(features)
  print "DBSCAN Labels:", labels[:10]
  sc = SpectralClustering()
  labels = sc.fit_predict(features)
  print "Spectral Labels:", labels
  km = KMeans(n_clusters=n/5)
  labels = km.fit_predict(features)
  print "KMeans Labels:", labels
  
  #code.interact()
  
  return labels

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
def plot_similarities(image_index, images, n_nearest, visual_tfidf, tags_tfidf, ext_tags_tfidf, gpses):
  def plot_sims(first_title, similarities):
    image_similarities = zip(similarities[image_index], range(similarities.shape[1]))
    nearest_pairs = sorted(image_similarities, key=lambda p: p[0], reverse=True)
    nearest_pairs = nearest_pairs[:n_nearest]
    nearest_indices = [pair[1] for pair in nearest_pairs]
    nearest_images = [images[i] for i in [pair[1] for pair in nearest_pairs]]
    nearest_sims = [pair[0] for pair in nearest_pairs]
    Utilities.plot_image_similarities(first_title, nearest_images, nearest_sims)
  # Plot by similarity of visual features
  similarities = cosine_similarity(visual_tfidf)
  #plot_sims('', similarities)  
  # Plot by similarity of tags
  similarities = cosine_similarity(tags_tfidf)
  plot_sims(' '.join(images[image_index].tags), similarities)  
  # Plot by similarity of extended_tags
  similarities = cosine_similarity(ext_tags_tfidf)
  plot_sims(' '.join(images[image_index].extended_tags), similarities)  
    
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
    
def cluster_by_tags_and_gps(images, folder):
  #images = images[:1000]
  # Shuffle images
  n_images = len(images)
  n_codebook = 10000
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
  cluster_similarity(similarity_matrix)
  
  print "Clustering views"
  clusters = find_view_clusters(features)
  
  save_clusters(images, clusters, folder + '+' + str(n_codebook))
  
  # Plot similarities and images (really ugly code)
  eduskuntatalo = None
  tuomiokirkko = None
  n_nearest = 20 # how many nearest hits to show
  for img, i in zip(images, range(len(images))):
    if '15791343418_b8c738bf32_z' in img.image_path:  # image of eduskuntatalo
      eduskuntatalo = i
    elif '15991895575_3705685e6c_z' in img.image_path:  # helsingin tuomiokirkko
      tuomiokirkko = i
  if eduskuntatalo != None:
    plot_similarities(eduskuntatalo, images, n_nearest, visual_tfidf, tags_tfidf, ext_tags_tfidf, gpses)
  if tuomiokirkko != None:
    plot_similarities(tuomiokirkko, images, n_nearest, visual_tfidf, tags_tfidf, ext_tags_tfidf, gpses)
  
  #code.interact(local=locals())

def main():
  folder = Utilities.get_folder_argument()
  base_folder = '../' + folder + '/'
  
  # Read metadata from files
  (image_paths, metadata_paths) = Utilities.get_image_paths(base_folder)
  images = []
  for i in range(len(image_paths)):
    images.append(Image(image_paths[i], metadata_paths[i]))
  
  # Start the main algorithm
  cluster_by_tags_and_gps(images, folder)

if __name__ == '__main__':
  main()