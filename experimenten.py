import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np
from PIL import Image
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint 
import pandas as pd
from pathlib import Path
from IPython.display import display, clear_output
import os

NUM_CLASSES = 3
TRAINPATH = 'dataset/train/'
VALIDPATH = 'dataset/valid/'
TESTPATH = 'dataset/test/'
NEWPATH = 'dataset/new/'

def CraeteLabelMap(path):
    annotations = pd.read_csv(path + 'annotation.csv')
    # Map label strings to integer labels
    labelMap = {}
    unique_labels = set(annotations['class'])
    for i, label in enumerate(unique_labels):
        labelMap[label] = i
    return labelMap
    
def readData(path, labelMap):
    target_size = (224, 224)
    annotations = pd.read_csv(path + 'annotation.csv')

    X_img = []
    y_labels = []
    y_bboxes = []

    for i, annotation in annotations.iterrows():
        image_path = annotation['filename']
        labels = annotation['class']
        xmin = annotation['xmin']
        ymin = annotation['ymin']
        xmax = annotation['xmax']
        ymax = annotation['ymax']

        # Preprocess the image
        image = Image.open(path + image_path)
        image = image.resize(target_size)
        image_data = np.array(image, dtype=np.float32)
        image_data /= 255.0

        # Normalize bounding box coordinates
        height, width, channels = image_data.shape
        xmin_norm = xmin / width
        ymin_norm = ymin / height
        xmax_norm = xmax / width
        ymax_norm = ymax / height
        
        X_img.append(image_data)
        y_labels.append(labelMap[labels])
        y_bboxes.append([xmin_norm, ymin_norm, xmax_norm, ymax_norm])

    X_img = np.array(X_img)
    y_labels = np.array(y_labels)
    y_bboxes = np.array(y_bboxes)
    
    # Convert labels to one-hot encoding
    Labels_cls = to_categorical(y_labels, NUM_CLASSES)

    return X_img, Labels_cls, y_bboxes

labelMap = CraeteLabelMap(TRAINPATH)

trainImg, trainLabels, trainbboxes = readData(TRAINPATH, labelMap)
validImg, validLabels, validbboxes = readData(VALIDPATH, labelMap)
testImg, testLabels, testbboxes = readData(TESTPATH, labelMap)

print("Train images shape:", trainImg.shape)
print("Train classes shape:", trainLabels.shape)
print("Train boxes shape:", trainbboxes.shape)

def augmentImages(images, labels, bboxes):
    augmentedImages = []
    augmentedLabels = []
    augmentedBboxes = []

    for img, label, bbox in zip(images, labels, bboxes):
        # Original image
        augmentedImages.append(img)
        augmentedLabels.append(label)
        augmentedBboxes.append(bbox)
        
        # Grayscale image
        gray_img = np.mean(img, axis=2, keepdims=True)
        gray_img = np.repeat(gray_img, 3, axis=2)
        augmentedImages.append(gray_img)
        augmentedLabels.append(label)
        augmentedBboxes.append(bbox)

        # # Inverted image
        # inverted_img = 1.0 - img
        # augmentedImages.append(inverted_img)
        # augmentedLabels.append(label)
        # augmentedBboxes.append(bbox)
        
        # Rotate the image by 90 degrees
        rotated_img = np.array(Image.fromarray((img * 255).astype(np.uint8)).rotate(90)) / 255.0
        augmentedImages.append(rotated_img)
        augmentedLabels.append(label)

        # Adjust bounding box coordinates
        # Original bbox coordinates
        xmin, ymin, xmax, ymax = bbox

        # Swap and adjust coordinates for 90-degree rotation
        new_xmin = ymin  # New xmin = old ymin
        new_ymin = 1 - xmax  # New ymin = 1 - old xmax
        new_xmax = ymax  # New xmax = old ymax
        new_ymax = 1 - xmin  # New ymax = 1 - old xmin

        # Store adjusted bbox
        augmentedBboxes.append([new_xmin, new_ymin, new_xmax, new_ymax])

    return np.array(augmentedImages), np.array(augmentedLabels), np.array(augmentedBboxes)

augmentedTrainImg, augmentedTrainLabels, augmentedTrainbboxes = augmentImages(trainImg, trainLabels, trainbboxes)
augmentedValidImg, augmentedValidLabels, augmentedValidbboxes = augmentImages(validImg, validLabels, validbboxes)
augmentedTestImg, augmentedTestLabels, augmentedTestbboxes = augmentImages(testImg, testLabels, testbboxes)

print("Train images shape:", augmentedTrainImg.shape)
print("Train classes shape:", augmentedTrainLabels.shape)
print("Train boxes shape:", augmentedTrainbboxes.shape)


def getModel():
    preModel = Path("models/softmax.h5")
    if preModel.is_file():
        # load trained model
        model = load_model('models/softmax.h5')
        return model
    else:
        # Define input tensor
        inputs = Input(shape=(224, 224, 3))
        
        # Convolutional layers
        x = Conv2D(32, kernel_size=(3, 3), activation="relu")(inputs)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Conv2D(64, kernel_size=(3, 3), activation="relu")(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Conv2D(128, kernel_size=(3, 3), activation="relu")(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        
        # Define the classification prediction head
        classification = Flatten()(x)
        classification = Dense(256, activation="relu")(classification)
        classification = Dropout(0.5)(classification)
        class_output = Dense(NUM_CLASSES, activation="softmax", name="class_output")(classification)
        
        # Define the bounding box prediction head
        bbox = Flatten()(x)
        bbox = Dense(256, activation="relu")(bbox)
        bbox = Dropout(0.5)(bbox)
        bbox_output = Dense(4, activation="linear", name="bbox_output")(bbox)
        
        
        # Define the model with input and output layers
        model = Model(inputs=inputs, outputs=(class_output, bbox_output))
        
        losses = {
            'class_output': 'categorical_crossentropy',
        	'bbox_output': 'mean_squared_error'
        }
        
        metrics={
            'class_output': 'accuracy',
            'bbox_output': 'accuracy'
        }
        
        trainTargets = {
        	"class_output": trainLabels,
        	"bbox_output": trainbboxes
            # "class_output": augmentedTrainLabels,
        	# "bbox_output": augmentedTrainbboxes
        }
        validTargets = {
        	"class_output": validLabels,
        	"bbox_output": validbboxes
            # "class_output": augmentedValidLabels,
        	# "bbox_output": augmentedValidbboxes
        }
        
        # Compile the model with categorical_crossentropy, mean_squared_error and Adam optimizer
        model.compile(loss=losses, optimizer='adam', metrics=metrics)
        
        # Train the model on the training data and validation on validation data
        checkpoint = ModelCheckpoint('models/softmax.h5', monitor='bbox_output_accuracy', save_best_only=True, mode='max', verbose=1)
        model.fit(trainImg, trainTargets, epochs=150, batch_size=50, validation_data=(validImg, validTargets), callbacks=[checkpoint])
        # model.fit(augmentedTrainImg, trainTargets, epochs=150, batch_size=16, validation_data=(augmentedValidImg, validTargets), callbacks=[checkpoint])
        
        return model


def testModel(model):
    testTargets = {
        "class_output": testLabels,
        "bbox_output": testbboxes
        # "class_output": augmentedTestLabels,
        # "bbox_output": augmentedTestbboxes
    }

    # Evaluate the model on the test set
    test_loss, test_bbox_loss, test_class_loss, test_bbox_acc, test_class_acc = model.evaluate(testImg, testTargets, verbose=0)

    # Print the test set metrics
    print('Test set loss: ', test_loss)
    print('Test set bounding box loss: ', test_bbox_loss)
    print('Test set class loss: ', test_class_loss)
    print('Test set bounding box accuracy: ', test_bbox_acc)
    print('Test set class accuracy: ', test_class_acc)


model = getModel()
testModel(model)


def unnormalize_bbox(bbox, origSize, resSize):
    scale_x = origSize[0] / resSize
    scale_y = origSize[1] / resSize

    xNorm, yNorm, wNorm, hNorm = bbox
    xmin = xNorm * (origSize[0] / scale_x)
    ymin = yNorm * (origSize[1] / scale_y)
    xmax = wNorm * (origSize[0] / scale_x)
    ymax = hNorm * (origSize[1] / scale_y)

    return (xmin, ymin, xmax, ymax)

def scaleBbox(bbox, origSize, resSize):
    # Calculate scaling factor
    scaleX = resSize[0] / origSize[0]
    scaleY = resSize[1] / origSize[1]

    xmin, ymin, xmax, ymax = bbox
    xminScaled = xmin * scaleX
    yminScaled = ymin * scaleY
    xmaxScaled = xmax * scaleX
    ymaxScaled = ymax * scaleY

    return (xminScaled, yminScaled, xmaxScaled, ymaxScaled)

def readNewImages(image_dir):
    target_size = (224, 224)
    X_img = []

    # List all image files
    image_files = [file for file in os.listdir(image_dir) if file.endswith('.jpg')]
    
    # Shuffle the list randomly
    np.random.shuffle(image_files)

    for image_file in image_files:
        image = Image.open(os.path.join(image_dir, image_file))
        image = image.resize(target_size)
        image_data = np.array(image, dtype=np.float32)
        image_data /= 255.0

        X_img.append(image_data)

    X_img = np.array(X_img)
    return X_img

# Load the new data
newImg = readNewImages(NEWPATH)


def prediction(img, labelMap):
    # Predict bounding boxes on the new data
    pred_labels, pred_bboxes = model.predict(img)

    orig_size = (1920, 1200)
    res_size = 224
    bboxes = [unnormalize_bbox(bbox, orig_size, res_size) for bbox in pred_bboxes]
    img_size = (224, 224)
    bboxScaled = [scaleBbox(bbox, orig_size, img_size) for bbox in bboxes]

    labelMap = {v: k for k, v in labelMap.items()}

    # Generate plot objects for each image and display them one by one
    for i in range(len(img)):
        fig, ax = plt.subplots()

        # Ensure the image has the correct shape
        current_img = img[i]
        
        if current_img.shape == (3,):  # If it has shape (3,), assume grayscale and reshape
            current_img = current_img.reshape(current_img.shape + (1,))

        lbl = np.argmax(pred_labels[i], axis=-1)
        ax.imshow(current_img)  # Use the reshaped image
        ax.axis('off')
        label_name = labelMap[lbl] 
        xmin, ymin, xmax, ymax = bboxScaled[i]
        ax.text(xmin, ymin, label_name, fontsize=10, color='red')
        rect = patches.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, fill=False, edgecolor='r')
        ax.add_patch(rect)

        # Display the plot in the notebook
        display(fig)
        plt.close(fig)

        # Wait for user input (press Enter) to continue to the next image
        input("Press Enter to continue to the next image...")
        clear_output(wait=True)

# Call the function
prediction(newImg, labelMap)