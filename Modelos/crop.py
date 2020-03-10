import numpy as np
import cv2
from sklearn.cluster import KMeans

def crop_cv(img,x,y):
    h = img.shape[0]
    w = img.shape[1]
    h0 = x
    h1 = h - x
    w0 = y
    w1 = w - y
    return img[h0:h1,w0:w1]

def crop_cv_pct(img, pct=0.25):
    h = img.shape[0]
    w = img.shape[1]
    x = int(pct * h)
    y = int(pct * w)
    h0 = x
    h1 = h - x
    w0 = y
    w1 = w - y
    return img[h0:h1,w0:w1]

def get_max_colors_(cluster, centroids):
    # Get the number of different clusters, create histogram, and normalize
    labels = np.arange(0, len(np.unique(cluster.labels_)) + 1)
    (hist, _) = np.histogram(cluster.labels_, bins = labels)
    hist = hist.astype("float")
    hist /= hist.sum()

    # Create frequency rect and iterate through each cluster's color and percentage
    rect = np.zeros((50, 300, 3), dtype=np.uint8)
    colors = sorted([(percent, color) for (percent, color) in zip(hist, centroids)])
    start = 0
    return max(colors)


def cluster_color(img):
    reshape = img.reshape((img.shape[0] * img.shape[1], 3))
    cluster = KMeans(n_clusters=5).fit(reshape)
    color_img = get_max_colors_(cluster, cluster.cluster_centers_)[1]
    return color_img
