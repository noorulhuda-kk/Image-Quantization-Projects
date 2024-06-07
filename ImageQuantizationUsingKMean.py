#Image Quantization using Python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from kneed import KneeLocator
img = plt.imread("hehe.jpg")
print("img shape (width, height, color channels): ",img.shape) #output: (981, 736, 3)
height, width, color_channels = img.shape

plt.imshow(img)
plt.show()

img = img.reshape(-1, 3)
# print(img)
print("length of img: ",len(img))
print("total unique colors: ", len(np.unique(img, axis=0)))
optimal_k=8

#since there is no elbow method, so i am randomly instructing 8 clusters from human eye perspective

kmeans = KMeans(n_clusters=8, random_state=42, n_init=10) 
print("labels: ",labels)
palette = kmeans.cluster_centers_.astype(int)
print("palette: ",palette)
quantized_img = palette[labels].reshape(height, width, color_channels)
print("quantized_img: ", quantized_img)
plt.imshow(quantized_img)
plt.show()
