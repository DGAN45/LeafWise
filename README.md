# LeafWise

LeafWise is a Streamlit-based plant leaf disease detection application for
diploma and academic demonstration use. It uses an existing TensorFlow/Keras
MobileNetV2 transfer-learning model to classify uploaded leaf images among 38
plant disease and healthy-leaf categories.

LeafWise provides model predictions with general plant-health information.
Prediction confidence is not a guaranteed diagnosis and should not replace
inspection or advice from a qualified agricultural professional.

## Features

- Upload JPG, JPEG, or PNG leaf images
- Validate file type, image readability, file size, and image dimensions
- Preview uploaded images
- Predict the top three likely classes
- Display the primary prediction, confidence, plant, disease or healthy state,
  visible symptoms, and general management suggestions
- Store successful predictions locally in CSV format
- View newest-first prediction history
- Clear prediction history after confirmation
- Display live prediction totals, healthy/diseased counts, and recent activity
- Explore healthy-versus-diseased results, frequent diseases, class distribution,
  and prediction trends with interactive charts
- Use a responsive green-and-white Streamlit interface

## Project Structure

```text
LeafWise/
├── app.py
├── config.py
├── requirements.txt
├── README.md
├── models/
│   ├── Plant_Disease_Model.keras
│   └── plant_leaf_class_names.json
├── utils/
│   ├── __init__.py
│   ├── predictor.py
│   ├── disease_info.py
│   ├── image_processing.py
│   └── history.py
├── assets/
│   └── logo.png                    # Optional placeholder
└── data/
    └── prediction_history.csv
```

The original `.keras` model and class JSON files were copied without changing
their contents or class order.

## Model Information

- Architecture: MobileNetV2 transfer learning
- Input shape: `(224, 224, 3)`
- Output shape: `(38,)`
- Output activation: softmax
- Supported classes: 38
- Trained model file: `models/Plant_Disease_Model.keras`
- Class labels: `models/plant_leaf_class_names.json`

The model contains a `Rescaling` preprocessing layer with:

- Scale: `0.00784313725490196`
- Offset: `-1.0`

The application therefore resizes uploaded images to `224 × 224` and converts
them to RGB float32 arrays without adding a second `[0, 1]` or `[-1, 1]`
rescaling step.

## Requirements

- Windows or a compatible Python environment
- Python 3.10 or newer recommended
- TensorFlow/Keras
- Streamlit
- Pillow
- NumPy
- Pandas
- Plotly

## Installation

Open a terminal in the `LeafWise` directory.

### 1. Create a virtual environment

```powershell
python -m venv .venv
```

### 2. Activate the environment

```powershell
.\\.venv\\Scripts\\Activate.ps1
```

If PowerShell blocks activation, a command-terminal alternative is:

```powershell
.\\.venv\\Scripts\\activate.bat
```

### 3. Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Run the application

```powershell
python -m streamlit run app.py
```

Open the local URL shown by Streamlit, normally:

```text
http://localhost:8501
```

## Usage

1. Open **Disease Detection**.
2. Upload a JPG, JPEG, or PNG leaf image no larger than 10 MB.
3. Review the image preview.
4. Select **Analyze leaf**.
5. Review the primary result and top three probabilities.
6. Read the symptom summary and general guidance.
7. Open **Prediction History** to review saved predictions.
8. Open **Analytics** to view aggregate prediction statistics.

## Module Responsibilities

### `app.py`

Contains the Streamlit interface and navigation. It connects image processing,
prediction, disease information, and history behavior.

### `config.py`

Centralizes model paths, input shape, class count, allowed formats, and the
history file location.

### `utils/predictor.py`

Loads and caches the Keras model and JSON class names. It validates model shapes,
class count, prediction dimensions, and top-three output.

### `utils/image_processing.py`

Validates uploads and converts readable RGB images into model-ready tensors.

### `utils/disease_info.py`

Separates model labels into plant and disease names and provides cautious,
general information for supported classes.

### `utils/history.py`

Creates and manages local timestamped prediction records in CSV format.

## Important Technical Decisions

- The existing trained model and class order are never modified or retrained.
- The model is cached after the first load to reduce repeated startup cost.
- Class labels are loaded directly from JSON in their original order.
- Uploaded bytes are converted to RGB before resizing.
- The existing Rescaling layer is not duplicated in application code.
- Prediction history stores metadata and confidence, not uploaded image files.
- The interface avoids guaranteed-diagnosis language and recommends expert
  verification.

## Prediction History Storage

Predictions are stored in:

```text
data/prediction_history.csv
```

Columns:

- `timestamp`
- `plant_name`
- `disease_name`
- `predicted_class`
- `confidence`

Use the confirmation checkbox in the Prediction History page before clearing
all records. Images are not stored, so history contains prediction metadata and
does not provide thumbnails.

## Testing Notes

The implementation has been tested for:

- Model loading and input/output shape validation
- 38-class JSON and model alignment
- Valid image conversion and synthetic model inference
- Top-three prediction structure
- Unsupported and corrupt image rejection
- Invalid prediction batch shape rejection
- Disease-information coverage
- History append and clear operations
- Streamlit startup and HTTP response

Test data or screenshots may be added by the project developer. Browser-level
visual testing with real leaf photographs is still recommended.

## Limitations

- The model is a machine-learning classifier; confidence is not certainty.
- Predictions may be inaccurate for unusual lighting, blur, backgrounds,
  damaged sensors, or plant classes outside the 38 training categories.
- Disease information is general and not a laboratory diagnosis.
- General crop-management information should be adapted to local regulations,
  crop variety, climate, and expert guidance.
- No pesticide or chemical dose recommendations are provided.
- Local CSV history is intended for a single demonstration deployment, not
  enterprise multi-user storage.

## Screenshot Placeholders

Add screenshots before submitting or presenting the project:

### Home / Dashboard

```text
![LeafWise Home Dashboard](assets/screenshots/home-dashboard.png)
```

### Disease Detection

```text
![LeafWise Disease Detection](assets/screenshots/disease-detection.png)
```

### Prediction History

```text
![LeafWise Prediction History](assets/screenshots/prediction-history.png)
```

### Analytics

```text
![LeafWise Analytics Dashboard](assets/screenshots/analytics.png)
```
