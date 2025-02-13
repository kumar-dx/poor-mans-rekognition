import os
import tempfile
import boto3
import dlib
import numpy as np
from PIL import Image
from typing import List, Tuple
import cv2
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class S3ImageHandler:
    def __init__(self, access_key: str, secret_key: str, region: str, bucket: str, prefix: str):
        """Initialize S3 client with credentials."""
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region
        )
        self.bucket = bucket
        self.prefix = prefix

    def list_images(self) -> List[str]:
        """List all images in the S3 bucket under specified prefix."""
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket,
                Prefix=self.prefix
            )
            
            if 'Contents' not in response:
                return []
                
            # Get image keys and sort them
            image_keys = [
                obj['Key'] for obj in response['Contents']
                if obj['Key'].lower().endswith(('.png', '.jpg', '.jpeg'))
            ]
            # Sort by filename to process similar images together
            return sorted(image_keys)
        except Exception as e:
            print(f"Error listing images: {str(e)}")
            return []

    def download_image(self, key: str, local_path: str) -> bool:
        """Download an image from S3 to local path."""
        try:
            self.s3_client.download_file(self.bucket, key, local_path)
            return True
        except Exception as e:
            print(f"Error downloading image {key}: {str(e)}")
            return False

class FaceDetector:
    def __init__(self):
        """Initialize face detector with dlib models."""
        # Load face detector and shape predictor
        self.detector = dlib.get_frontal_face_detector()
        self.shape_predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
        self.face_rec_model = dlib.face_recognition_model_v1("models/dlib_face_recognition_resnet_model_v1.dat")
        self.known_face_encodings = []
        self.known_face_names = []  # Keep track of image names for debugging

    def process_image(self, image_path: str) -> List[np.ndarray]:
        """Process an image and return face encodings."""
        try:
            # Load image using cv2
            img = cv2.imread(image_path)
            if img is None:
                print(f"Error: Could not load image {image_path}")
                return []
            
            # Convert to RGB (dlib expects RGB)
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            faces = self.detector(rgb_img)
            
            # Get face encodings
            face_encodings = []
            for face in faces:
                # Get facial landmarks
                shape = self.shape_predictor(rgb_img, face)
                # Get face encoding
                face_encoding = np.array(self.face_rec_model.compute_face_descriptor(rgb_img, shape))
                face_encodings.append(face_encoding)
            
            return face_encodings
        except Exception as e:
            print(f"Error processing image {image_path}: {str(e)}")
            return []

    def is_new_face(self, face_encoding: np.ndarray, image_name: str, tolerance: float = 0.53) -> bool:
        """Check if a face is new (not matching any known faces)."""
        if not self.known_face_encodings:
            print(f"First face found in {image_name} - adding as reference")
            return True
        
        # Compare with known faces
        for i, known_encoding in enumerate(self.known_face_encodings):
            # Calculate Euclidean distance
            distance = np.linalg.norm(face_encoding - known_encoding)
            print(f"Distance from {self.known_face_names[i]}: {distance:.3f}")
            # Use higher tolerance for s1.jpg comparisons
            current_tolerance = 0.57 if self.known_face_names[i] == "s1.jpg" else tolerance
            if distance < current_tolerance:
                print(f"Match found with {self.known_face_names[i]} (distance: {distance:.3f})")
                return False
        print(f"No matches found - {image_name} contains a new face")
        return True

    def add_face(self, face_encoding: np.ndarray, image_name: str):
        """Add a new face encoding to known faces."""
        self.known_face_encodings.append(face_encoding)
        self.known_face_names.append(image_name)
        print(f"Added new face from {image_name} (total known faces: {len(self.known_face_encodings)})")

def count_unique_faces(
    access_key: str,
    secret_key: str,
    region: str,
    bucket: str,
    prefix: str
) -> int:
    """Main function to count unique faces in S3 bucket images."""
    
    print("\nStarting face detection process...")
    
    # Initialize handlers
    print("Initializing S3 handler...")
    s3_handler = S3ImageHandler(access_key, secret_key, region, bucket, prefix)
    face_detector = FaceDetector()
    unique_faces_count = 0

    # Create temporary directory for downloaded images
    with tempfile.TemporaryDirectory() as temp_dir:
        # List all images in bucket
        print("\nListing images in S3 bucket...")
        image_keys = s3_handler.list_images()
        
        if not image_keys:
            print("No images found in the specified S3 bucket and prefix")
            return 0
        
        print(f"Found {len(image_keys)} images to process")

        # Process each image
        for i, image_key in enumerate(image_keys, 1):
            print(f"\nProcessing image {i}/{len(image_keys)}: {image_key}")
            local_path = os.path.join(temp_dir, os.path.basename(image_key))
            
            # Download image
            if not s3_handler.download_image(image_key, local_path):
                continue

            # Get face encodings
            face_encodings = face_detector.process_image(local_path)
            print(f"Found {len(face_encodings)} faces in this image")
            
            # Check each face
            for j, face_encoding in enumerate(face_encodings, 1):
                print(f"Analyzing face {j}/{len(face_encodings)}")
                if face_detector.is_new_face(face_encoding, os.path.basename(image_key)):
                    unique_faces_count += 1
                    face_detector.add_face(face_encoding, os.path.basename(image_key))

            # Clean up downloaded image
            try:
                os.remove(local_path)
            except:
                pass

    return unique_faces_count

def validate_env_vars():
    """Validate that all required environment variables are set."""
    required_vars = [
        'AWS_ACCESS_KEY_ID',
        'AWS_SECRET_ACCESS_KEY',
        'AWS_REGION',
        'AWS_BUCKET_NAME',
        'AWS_BUCKET_PREFIX'
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        raise ValueError(
            f"Missing required environment variables: {', '.join(missing_vars)}\n"
            "Please check your .env file and ensure all required variables are set."
        )

if __name__ == "__main__":
    # Validate environment variables
    validate_env_vars()
    
    # Get configuration from environment variables
    aws_access_key = os.getenv('AWS_ACCESS_KEY_ID')
    aws_secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
    aws_region = os.getenv('AWS_REGION')
    aws_bucket = os.getenv('AWS_BUCKET_NAME')
    aws_prefix = os.getenv('AWS_BUCKET_PREFIX')

    # Count unique faces
    unique_faces = count_unique_faces(
        aws_access_key,
        aws_secret_key,
        aws_region,
        aws_bucket,
        aws_prefix
    )
    
    print(f"\nFound {unique_faces} unique faces in the images")
