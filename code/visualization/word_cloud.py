'''
Created on 2016

@author: Graham Reid

I imagine that this will be the thing that people are most interested in. It is
just a super simple example of building a word cloud using Andreas Mueller's
word cloud generating code which can be found at:

https://github.com/amueller/word_cloud
'''

import pickle
import numpy as np
from PIL import Image
from os import path
import matplotlib.pyplot as plt
import random

from wordcloud import WordCloud, STOPWORDS

amp_factor = 100

feature_location = '../../data/weighted_words/weighted_words.pk'
mask_location = 'mask_2.jpg'

best_bag_array = []
best_bag_weight_array = []

worst_bag_array = []
worst_bag_weight_array = []

with open(feature_location, 'rb') as input:
    best_bag_array = pickle.load(input)
    best_bag_weight_array = pickle.load(input)
    worst_bag_array = pickle.load(input)
    worst_bag_weight_array = pickle.load(input)

best_text = ""
worst_text = ""
combined_text = ""

for i in xrange(0,len(best_bag_array)) :
    best_text = best_text + (' ' + best_bag_array[i].replace(" ", "_"))* \
    int(np.log(np.abs(best_bag_weight_array[i])*amp_factor))

    combined_text = combined_text + (' ' + best_bag_array[i])* \
    int(np.log(np.abs(best_bag_weight_array[i])*amp_factor))

for i in xrange(0,len(worst_bag_array)) :
    worst_text = worst_text + (' ' + worst_bag_array[i])* \
    int(np.abs(worst_bag_weight_array[i])*amp_factor)

    combined_text = combined_text + (' ' + best_bag_array[i])* \
    int(np.abs(best_bag_weight_array[i])*amp_factor)

'''
probably a better way of doing this, but my jpeg had some non black/white pixels
so I converted them to being fully black and white
'''
image = np.array(Image.open(mask_location))
shape = image.shape
image = np.reshape(image, -1)
image[image > 0] = 255
mask = np.reshape(image, shape)

def grey_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    return "hsl(0, 0%%, %d%%)" % random.randint(0, 30)

wc = WordCloud(mask=mask,background_color="white", \
max_font_size=80, relative_scaling = 0.5, max_words=1000, random_state=1).generate(worst_text)
#store default colored image
default_colors = wc.to_array()
plt.figure()
plt.imshow(wc.recolor(color_func=grey_color_func, random_state=3))
#plt.imshow(default_colors)
plt.axis("off")
plt.show()

wc.to_file("worst_suit.png")
