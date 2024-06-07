#KNN approach
import numpy as np
import matplotlib.pyplot as plt
from tkinter import colorchooser
from sklearn.neighbors import KNeighborsClassifier
# from sklearn.cluster import KMeans
# from kneed import KneeLocator

img = plt.imread("hehe.jpg") #read file
print("img shape (width, height, color channels): ",img.shape) #output: (981, 736, 3)
height, width, color_channels = img.shape

plt.imshow(img)
# plt.show()

img = img.reshape(-1, 3) #reshape in2 2D array of pixels
# print(img)
print("length of img: ",len(img))
print("total unique colors: ", len(np.unique(img, axis=0)))
# optimal_k=8

#taking colrs by user
k=int(input("How many total colors do you want to use? "))
given_rgb_colors=[]
for i in range(k):
    # open color chooser dialog
    color = colorchooser.askcolor()
    # get RGB values from the selected color
    rgb = color[0]
    given_rgb_colors.append(rgb)
    # print RGB values
    print("RGB:", rgb)
print(given_rgb_colors)

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(np.array(given_rgb_colors), np.arange(k))

labels = knn.predict(img)
quantized_img = np.array(given_rgb_colors)[labels].reshape(height, width, color_channels)

plt.imshow(quantized_img)
plt.show()

# print(type(palette))
rgbPallete = np.array(given_rgb_colors)
print(type(given_rgb_colors))
print(rgbPallete)
