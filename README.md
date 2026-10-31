# Traffic Sign Classification using MobileNetV2

A deep-learning project that classifies traffic-sign images into their
corresponding categories using **MobileNetV2 transfer learning** with
TensorFlow/Keras. The project includes image preprocessing, data
augmentation, class-imbalance handling, model fine-tuning, evaluation,
and an interactive image-upload prediction interface.

## Project Overview

The model accepts an image containing a traffic sign and predicts its
class. It uses an ImageNet-pretrained MobileNetV2 backbone as a feature
extractor, followed by a custom classification head.

### Highlights

-   MobileNetV2 with ImageNet pretrained weights
-   Input images resized to **96 × 96 RGB**
-   Data augmentation with random rotation, zoom, and translation
-   Pixel rescaling to the range expected by MobileNetV2 (`[-1, 1]`)
-   Stratified train, validation, and test splits
-   Balanced class weights to help address class imbalance
-   Two-stage training: classifier-head training followed by fine-tuning
-   Checkpointing, early stopping, and learning-rate reduction
-   Classification report, accuracy/loss plots, and confusion matrix
-   Interactive upload widget for predicting one or more images

## Dataset

This project uses the [Traffic Sign Dataset --
Classification](https://www.kaggle.com/datasets/ahemateja19bec1025/traffic-sign-dataset-classification)
from Kaggle.

The dataset is organized into class-specific image folders and includes
a `labels.csv` file mapping class IDs to traffic-sign names. Public
dataset summaries describe approximately 58 traffic-sign classes; the
exact number of images/classes available to this notebook can vary with
the downloaded dataset version and the notebook's filtering.

### Expected dataset layout

The training notebook searches the Kaggle input directory for
`traffic_Data/DATA` and `labels.csv`. The expected structure is:

``` text
traffic-sign-dataset-classification/
├── labels.csv
└── traffic_Data/
    ├── DATA/
    │   ├── 0/
    │   │   ├── image_1.jpg
    │   │   └── ...
    │   ├── 1/
    │   │   └── ...
    │   └── ...
    └── TEST/
        └── ...
```

Each numeric folder under `DATA/` represents a class ID. The class names
are read from `labels.csv`, which is expected to contain `ClassId` and
`Name` columns.

> **Note:** The current notebook loads images from `traffic_Data/DATA`.
> It does not directly evaluate the separate `traffic_Data/TEST` folder.
> Instead, it creates its own stratified train/validation/test split
> from the images loaded from `DATA`.

## Model Architecture

The model pipeline is:

1.  Input RGB image (`96 × 96 × 3`)
2.  Data augmentation:
    -   Random rotation: `0.04`
    -   Random zoom: `0.10`
    -   Random translation: `0.05` horizontally and vertically
3.  Rescaling from `[0, 255]` to `[-1, 1]`
4.  MobileNetV2 convolutional base with ImageNet weights and no top
    classifier
5.  Global average pooling
6.  Dropout (`0.35`)
7.  Dense layer with 128 units and ReLU activation
8.  Dropout (`0.25`)
9.  Softmax output layer with one output per retained class

### Training strategy

**Stage 1 -- Feature extraction**

-   MobileNetV2 backbone frozen
-   Adam optimizer, learning rate `0.001`
-   Sparse categorical cross-entropy loss
-   Up to 15 epochs
-   Batch size: 32

**Stage 2 -- Fine-tuning**

-   Last 30 MobileNetV2 layers made trainable
-   Batch-normalization layers kept frozen
-   Adam optimizer, learning rate `0.00001`
-   Up to 12 epochs
-   Batch size: 32

Both stages use class weights and the same callbacks:

-   Model checkpoint monitoring validation loss
-   Early stopping with patience of 5 and best-weight restoration
-   Reduce learning rate on validation-loss plateau (factor `0.3`,
    patience `2`, minimum learning rate `1e-7`)

The notebook sets the random seed to `42`.

## Data Preparation

-   Supported image formats: JPG, JPEG, PNG, BMP, and WEBP
-   Images are converted to RGB and resized to `96 × 96`
-   Classes with fewer than **6 successfully loaded images** are
    excluded
-   Remaining class IDs are remapped to contiguous output indices for
    model training
-   A stratified split is performed:
    -   70% training
    -   15% validation
    -   15% testing

The saved class-mapping CSV preserves the relationship between model
output indices, original dataset class IDs, and class names.

## Evaluation

After training, the best saved checkpoint is loaded and evaluated on the
held-out test split. The notebook reports:

-   Test loss
-   Test accuracy
-   Per-class precision, recall, and F1-score
-   Accuracy and loss curves for training and validation
-   Confusion matrix

**Results:** No fixed accuracy or metric values are stated here because
results depend on the dataset files, runtime, and training execution.
Run the notebook to generate the metrics for your experiment.

## Repository Structure

A suggested GitHub repository layout:

``` text
.
├── traffic-sign.ipynb
├── README.md
├── requirements.txt
└── .gitignore
```

The trained model and class mapping are generated in the Kaggle working
directory:

``` text
/kaggle/working/
├── traffic_sign_classifier.keras
├── traffic_sign_class_mapping.csv
└── best_traffic_sign.keras
```

These generated files are not automatically included in the GitHub
repository. Download or copy them from the notebook environment if you
want to version or distribute them. Model files can be large; consider
Git LFS or a model-hosting service.

## Getting Started

### 1. Clone the repository

``` bash
git clone https://github.com/<your-username>/<your-repository>.git
cd <your-repository>
```

### 2. Install dependencies

Use Python 3.9 or later with compatible TensorFlow/Keras packages.

``` bash
pip install -r requirements.txt
```

Suggested `requirements.txt`:

``` text
tensorflow
numpy
pandas
matplotlib
seaborn
Pillow
scikit-learn
ipywidgets
jupyter
```

For reproducible experiments, pin package versions after confirming the
versions that work in your environment.

### 3. Download and attach the dataset

Download the dataset from
[Kaggle](https://www.kaggle.com/datasets/ahemateja19bec1025/traffic-sign-dataset-classification).

-   **Kaggle Notebook:** Add the dataset using the notebook's **Add
    Input** option. The code searches `/kaggle/input/` for
    `traffic_Data/DATA` and `labels.csv`.
-   **Local Jupyter environment:** Download and extract the dataset,
    then update the dataset discovery paths in the notebook to point to
    your local `traffic_Data/DATA` folder and `labels.csv`.

### 4. Run the notebook

Open `traffic-sign.ipynb` in Kaggle Notebooks or Jupyter and run the
cells from top to bottom.

The training cell will load the images, prepare labels, train and
fine-tune the model, evaluate the best checkpoint, display plots, and
save the model and class mapping.

## Making Predictions

### Prediction from a single image

In the training notebook, set `TEST_IMAGE` to the path of an image file:

``` python
TEST_IMAGE = "/path/to/your/traffic_sign.jpg"
```

Run the prediction section. The notebook displays the predicted
traffic-sign name, original class ID, and confidence score.

### Interactive image upload

After training, run the second code section in the notebook. It loads:

-   `traffic_sign_classifier.keras`
-   `traffic_sign_class_mapping.csv`

An upload widget will appear. Select one or more supported image files,
and the notebook will display each original image with its predicted
class, class ID, and confidence.

The interactive prediction section expects the model and mapping CSV at
the Kaggle paths shown in the notebook. If running locally, change
`MODEL_PATH` and `MAPPING_PATH` to their actual locations.

## Saved Artifacts

  -----------------------------------------------------------------------
  File                                Description
  ----------------------------------- -----------------------------------
  `traffic_sign_classifier.keras`     Final trained Keras model

  `traffic_sign_class_mapping.csv`    Mapping from output index to
                                      original class ID and class name

  `best_traffic_sign.keras`           Best checkpoint selected using
                                      validation loss
  -----------------------------------------------------------------------

Keep the class-mapping CSV with the model. The model's softmax output
index is not necessarily the original dataset class ID.

## Limitations

-   This is an **image classification** project: it predicts a class for
    an input image. It does not locate traffic signs in a full road
    scene or return bounding boxes.
-   The notebook excludes classes with fewer than six loaded images.
-   The held-out test split is created from `traffic_Data/DATA`; the
    dataset's separate `TEST` directory is not used by the current
    training code.
-   Predictions can be affected by image quality, lighting, angle,
    occlusion, background clutter, and differences between training
    images and real-world road conditions.
-   A confidence score is the model's softmax output; it should not
    automatically be interpreted as a calibrated probability.
-   This project is an educational/research prototype and should not be
    used as the sole basis for driving or road-safety decisions.

## Future Improvements

-   Evaluate against the dataset's separate test directory, if its
    labels and intended evaluation protocol are available.
-   Add top-k predictions and confidence calibration.
-   Compare MobileNetV2 with custom CNNs and other transfer-learning
    backbones.
-   Add Grad-CAM visualizations to inspect model attention.
-   Export an optimized model for mobile or edge inference.
-   Test robustness under varied lighting, blur, rotation, and
    occlusion.

## Acknowledgements

-   Dataset: [Traffic Sign Dataset -- Classification
    (Kaggle)](https://www.kaggle.com/datasets/ahemateja19bec1025/traffic-sign-dataset-classification)
-   Pretrained architecture:
    [MobileNetV2](https://keras.io/api/applications/mobilenet/#mobilenetv2-function)
-   Framework: [TensorFlow](https://www.tensorflow.org/) /
    [Keras](https://keras.io/)

Please refer to the Kaggle dataset page for the dataset's current
license and attribution requirements.

------------------------------------------------------------------------

If you use or extend this project, please cite the original dataset and
the relevant software/frameworks.
