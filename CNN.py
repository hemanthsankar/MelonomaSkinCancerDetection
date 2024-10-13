import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16, ResNet50, InceptionV3
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Load and preprocess the ISIC 2020 dataset
def load_images_from_folder(folder, label):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = tf.io.read_file(img_path)
        img = tf.image.decode_image(img, channels=3)
        img = tf.image.resize(img, (224, 224))
        img = img / 255.0
        images.append(img)
        labels.append(label)
    return images, labels

def load_data(data_dir):
    # Load the training images and labels
    train_benign_images, train_benign_labels = load_images_from_folder(os.path.join(data_dir, 'archive (2)', 'data', 'train', 'benign'), 0)
    train_malignant_images, train_malignant_labels = load_images_from_folder(os.path.join(data_dir, 'archive (2)', 'data', 'train', 'malignant'), 1)
    
    train_images = train_benign_images + train_malignant_images
    train_labels = train_benign_labels + train_malignant_labels
    
    # Load the validation images and labels
    val_benign_images, val_benign_labels = load_images_from_folder(os.path.join(data_dir, 'archive (2)', 'data', 'test', 'benign'), 0)
    val_malignant_images, val_malignant_labels = load_images_from_folder(os.path.join(data_dir, 'archive (2)', 'data', 'test', 'malignant'), 1)
    
    val_images = val_benign_images + val_malignant_images
    val_labels = val_benign_labels + val_malignant_labels
    
    return np.array(train_images), np.array(train_labels), np.array(val_images), np.array(val_labels)

# Implement baseline CNN models
def build_baseline_models():
    # VGG16 model
    vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    vgg16_model = Sequential()
    vgg16_model.add(vgg16)
    vgg16_model.add(Flatten())
    vgg16_model.add(Dense(512, activation='relu'))
    vgg16_model.add(Dropout(0.5))
    vgg16_model.add(Dense(1, activation='sigmoid'))
    
    # ResNet50 model
    resnet50 = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    resnet50_model = Sequential()
    resnet50_model.add(resnet50)
    resnet50_model.add(Flatten())
    resnet50_model.add(Dense(512, activation='relu'))
    resnet50_model.add(Dropout(0.5))
    resnet50_model.add(Dense(1, activation='sigmoid'))
    
    # Inception-v3 model
    inception_v3 = InceptionV3(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    inception_v3_model = Sequential()
    inception_v3_model.add(inception_v3)
    inception_v3_model.add(Flatten())
    inception_v3_model.add(Dense(512, activation='relu'))
    inception_v3_model.add(Dropout(0.5))
    inception_v3_model.add(Dense(1, activation='sigmoid'))
    
    return vgg16_model, resnet50_model, inception_v3_model

# Implement the proposed CNN model
def build_proposed_model():
    # Define the custom CNN architecture
    proposed_model = Sequential()
    proposed_model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
    proposed_model.add(tf.keras.layers.BatchNormalization())
    proposed_model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    proposed_model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    proposed_model.add(tf.keras.layers.BatchNormalization())
    proposed_model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    proposed_model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
    proposed_model.add(tf.keras.layers.BatchNormalization())
    proposed_model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    proposed_model.add(tf.keras.layers.Flatten())
    proposed_model.add(tf.keras.layers.Dense(512, activation='relu'))
    proposed_model.add(tf.keras.layers.Dropout(0.5))
    proposed_model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    
    return proposed_model

# Ensemble multiple CNN models
def ensemble_models(models, X_test, y_test):
    # Make predictions from each model
    predictions = [model.predict(X_test) for model in models]
    
    # Combine the predictions using majority voting
    ensemble_prediction = np.round(np.mean(predictions, axis=0))
    
    # Evaluate the ensemble model
    accuracy = accuracy_score(y_test, ensemble_prediction)
    precision = precision_score(y_test, ensemble_prediction)
    recall = recall_score(y_test, ensemble_prediction)
    f1 = f1_score(y_test, ensemble_prediction)
    auc_roc = roc_auc_score(y_test, ensemble_prediction)
    
    return accuracy, precision, recall, f1, auc_roc

# Main function
def main():
    # Load and preprocess the ISIC 2020 dataset
    data_dir = '/content/skin_cancer_detection_using_deeplearning'
    X_train, y_train, X_val, y_val = load_data(data_dir)
    
    # Build and train the baseline CNN models
    vgg16_model, resnet50_model, inception_v3_model = build_baseline_models()
    vgg16_model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    resnet50_model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    inception_v3_model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    
    # Train the baseline models
    vgg16_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50)
    resnet50_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50)
    inception_v3_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50)
    
    # Build and train the proposed CNN model
    proposed_model = build_proposed_model()
    proposed_model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    proposed_model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50)
    
    # Evaluate the baseline and proposed models on the validation set
    models = [vgg16_model, resnet50_model, inception_v3_model, proposed_model]
    accuracy, precision, recall, f1, auc_roc = ensemble_models(models, X_val, y_val)
    
    print(f'Ensemble Model Performance:')
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print(f'AUC-ROC: {auc_roc:.4f}')

if _name_ == '_main_':
    main()
