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

# Elbow method to find optimal K
distortions = []
K = range(1, 10)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(img)
    distortions.append(kmeans.inertia_)
    print(k)

# second_derivatives = np.gradient(np.gradient(distortions))[1:-1]
# optimal_k = np.argmin(second_derivatives) + 2
kneedle = KneeLocator(range(1, 10), distortions, curve='convex', direction='decreasing')
optimal_k = kneedle.knee
print("K should be: ",optimal_k)
#
plt.figure()
plt.plot(K, distortions, 'bx-')
plt.xlabel('K')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal K')
plt.show()

kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10) 
print("labels: ",labels)
palette = kmeans.cluster_centers_.astype(int)
print("palette: ",palette)
quantized_img = palette[labels].reshape(height, width, color_channels)
print("quantized_img: ", quantized_img)
plt.imshow(quantized_img)
plt.show()
