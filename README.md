# Face Counter for S3 Images

This script analyzes images stored in an AWS S3 bucket to count unique faces. It uses dlib for face detection and recognition, providing accurate results even with variations in facial expressions and angles.

## Prerequisites

- Python 3.x
- AWS S3 bucket with images
- AWS credentials with S3 access

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install boto3 dlib numpy Pillow opencv-python python-dotenv
```

4. Download required dlib models:
```bash
mkdir -p models
cd models
curl -L "https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat.bz2" -o shape_predictor_68_face_landmarks.dat.bz2
curl -L "https://github.com/davisking/dlib-models/raw/master/dlib_face_recognition_resnet_model_v1.dat.bz2" -o dlib_face_recognition_resnet_model_v1.dat.bz2
bunzip2 *.bz2
cd ..
```

## Configuration

1. Copy the example environment file:
```bash
cp .env.example .env
```

2. Edit `.env` file with your AWS credentials and S3 configuration:
```
AWS_ACCESS_KEY_ID=your_access_key_here
AWS_SECRET_ACCESS_KEY=your_secret_key_here
AWS_REGION=your_region_here
AWS_BUCKET_NAME=your_bucket_name_here
AWS_BUCKET_PREFIX=your_prefix_here
```

## Usage

Run the script:
```bash
python3 unique_faces_counter.py
```

The script will:
1. Connect to your S3 bucket
2. Download and process each image
3. Detect faces and compare them
4. Print the total count of unique faces found

## How it Works

- Uses dlib's face detection to find faces in images
- Generates face encodings for comparison
- Compares faces using Euclidean distance with optimized tolerance values
- Groups similar faces together to count unique individuals
- Handles variations in facial expressions and angles
- Processes images in sorted order for better grouping

## Output

The script provides detailed logging of:
- Image processing progress
- Face detection results
- Face comparison distances
- Final count of unique faces

## Note

Keep your `.env` file secure and never commit it to version control. The `.env.example` file is provided as a template.
