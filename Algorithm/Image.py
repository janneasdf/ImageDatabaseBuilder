import json
from dateutil import parser
from datetime import datetime

class Image:
  image_path = None
  metadata = None
  features = []
  tags = []
  extended_tags = []
  gps = []
  date = None
  
  def __init__(self, image_path, metadata_path):
    self.image_path = image_path
    self.load_metadata(metadata_path)
    
  def load_metadata(self, metadata_path):
    with open(metadata_path, 'r') as f:
      self.metadata = json.load(f)
    self.load_tags()
    gps = self.metadata['gps']
    self.gps = [float(gps[0]), float(gps[1])]
    date_string = self.metadata['datetaken']
    self.date = parser.parse(date_string)
    
  def load_tags(self):
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
    extended = [tag for tag in self.tags]
    extended = add_tags_from_text(extended, self.metadata['title'])
    extended = add_tags_from_text(extended, self.metadata['description'])
    self.extended_tags = extended
      