import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# Function to load and preprocess images
def load_images_from_folder(folder, image_size=(64, 64)):
    images = []
    labels = []
    for label in os.listdir(folder):
        label_folder = os.path.join(folder, label)
        if os.path.isdir(label_folder):
            for filename in os.listdir(label_folder):
                img_path = os.path.join(label_folder, filename)
                img = cv2.imread(img_path)
                if img is not None:
                    # Resize and normalize the image
                    img = cv2.resize(img, image_size)
                    img = img / 255.0  # Normalize the image
                    img = img.flatten()  # Flatten the image to a 1D array
                    images.append(img)
                    labels.append(label)
    return np.array(images), np.array(labels)

# Load dataset (adjust folder path accordingly)
data_folder = "path_to_dataset"  # Folder structure: data_folder/class1/, data_folder/class2/, etc.
X, y = load_images_from_folder(data_folder)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the KNN classifier
knn = KNeighborsClassifier(n_neighbors=3)  # You can adjust n_neighbors
knn.fit(X_train, y_train)

# Predict on the test set
y_pred = knn.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Optional: Visualize some test results
def visualize_results(X_test, y_test, y_pred, num_images=5):
    for i in range(num_images):
        img = X_test[i].reshape(64, 64, 3)  # Reshape back to image size
        label_true = y_test[i]
        label_pred = y_pred[i]
        cv2.imshow(f"True: {label_true}, Pred: {label_pred}", img)
        cv2.waitKey(0)
    cv2.destroyAllWindows()

# Visualize predictions for some test images
visualize_results(X_test, y_test, y_pred)
