# Skin Defect Detection

This repository contains a Gradio-based web application that detects and highlights skin defects such as redness or acne in face images. The application uses OpenCV, MediaPipe, and Gradio to detect faces, enhance contrast, and identify potential skin issues. This project is one way to try to solve the problem, but it can't be used for accurate skin issue detection yet. More work is needed to make the algorithm better at handling different situations.

## Features

**Face Detection**: Uses MediaPipe to detect face mesh  and draw bounding boxes around detected faces.

**Contrast Enhancement**: Applies histogram equalization and CLAHE to enhance the image contrast.

**Skin Defect Highlighting**: Uses the face mesh approach from MediaPipe to precisely detect facial regions, while removing unnecessary areas like the eyes and lips from the region of interest. Potential skin issues like redness are identified and highlighted in these areas for easier visualization.

**Web Interface**: A simple web interface built with Gradio to easily upload and process images.

## Installation

Clone the repository:

```bash
git clone git@github.com:bzavolovich/skin_condition_detector.git
```

Navigate to the project directory:

```bash
cd skin-defect-detection
```

Install the required dependencies:

```bash 
pip install -r requirements.txt
```

## Usage

Run the application:

```bash
python gradio_app.py
```

Open your browser and navigate to the provided local URL to use the web interface.

Upload an image.

All dependencies can be installed using the requirements.txt file.

## Examples

Upload an image to see the following results:

Detected faces with bounding boxes.

Highlighted areas indicating potential skin issues such as acne or redness.

This algorithm was tested on a single image (acne1.png) located in the root of the repository. Using it on other images may lead to inaccurate results, and the algorithm needs to be adapted for more general cases.

## links

[Mediapipe face_mesh scheme](https://github.com/google-ai-edge/mediapipe/blob/e0eef9791ebb84825197b49e09132d3643564ee2/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png)

