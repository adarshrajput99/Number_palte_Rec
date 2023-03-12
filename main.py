
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

# Load the dataset
dataset = []
labels = []
for i in range(1, 101):
    img = cv2.imread(f"dataset/{i}.jpg")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dataset.append(gray)
    labels.append(i % 10)

# Preprocess the images
processed_images = []
for img in dataset:
    processed_img = cv2.resize(img, (50, 50))
    processed_img = cv2.GaussianBlur(processed_img, (5, 5), 0)
    processed_img = cv2.threshold(processed_img, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
    processed_images.append(processed_img)

# Segment the characters
segmented_chars = []
for img in processed_images:
    contours, _ = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        if w >= 10 and h >= 10:
            roi = img[y:y+h, x:x+w]
            segmented_chars.append(roi)

# Extract features from the characters
features = []
for char in segmented_chars:
    hog_feature = cv2.HOGDescriptor((50, 50), (10, 10), (5, 5), (5, 5), 9).compute(char)
    features.append(hog_feature)

# Train the model
X_train = np.array(features).squeeze()
y_train = np.array(labels)
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_train)
print(confusion_matrix(y_train, y_pred))
print(classification_report(y_train, y_pred))


