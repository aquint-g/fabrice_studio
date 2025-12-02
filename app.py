import os
import time
import json
import requests
from flask import Flask, render_template, request, jsonify
from google.cloud import aiplatform
from google.cloud import storage
import vertexai
from vertexai.preview.generative_models import GenerativeModel, Part, Image
import google.auth

app = Flask(__name__)

# --- Configuration ---
# Replace these with your actual values or set them as environment variables
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "your-project-id")
REGION = os.environ.get("REGION", "us-central1")
BUCKET_NAME = os.environ.get("BUCKET_NAME", "veodemobucket")
VEO_API_KEY = os.environ.get("VEO_API_KEY","nokey")

# Initialize Vertex AI
try:
    vertexai.init(project=PROJECT_ID, location=REGION)
except Exception as e:
    print(f"Warning: Vertex AI init failed (expected if no creds): {e}")

# --- Helper Functions ---

def upload_to_gcs(source_file_path, destination_blob_name):
    """Uploads a file to the bucket."""
    try:
        storage_client = storage.Client(project=PROJECT_ID)
        bucket = storage_client.bucket(BUCKET_NAME)
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(source_file_path)
        return f"gs://{BUCKET_NAME}/{destination_blob_name}"
    except Exception as e:
        print(f"Error uploading to GCS: {e}")
        raise

def analyze_last_frame_for_coherence(local_image_path):
    """
    1. Uploads the image/video to GCS.
    2. Uses Gemini to describe it.
    3. Returns the GCS URI of the *image* (frame) for Veo.
    """
    import cv2
    
    filename = os.path.basename(local_image_path)
    ext = os.path.splitext(filename)[1].lower()
    
    veo_gcs_uri = None
    gemini_part = None
    
    # Fallback to gemini-2.0-flash-exp
    GEMINI_MODEL_ID = "gemini-2.5-flash" 

    try:
        if ext in ['.mp4', '.mov', '.avi', '.mkv']:
            # --- VIDEO HANDLING ---
            print(f"Detected video file: {filename}")
            
            # 1. Extract Last Frame for Veo
            cap = cv2.VideoCapture(local_image_path)
            if not cap.isOpened():
                print("Error: Could not open video.")
                raise ValueError("Could not open video file.")
            
            # Seek to end
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                print("Error: Could not read last frame.")
                raise ValueError("Could not extract last frame from video.")
            
            # Save frame locally
            frame_filename = f"frame_{filename}.jpg"
            frame_path = os.path.join("uploads", frame_filename)
            cv2.imwrite(frame_path, frame)
            print(f"Extracted last frame to {frame_path}")
            
            # Upload Frame to GCS for Veo
            veo_gcs_uri = upload_to_gcs(frame_path, f"uploads/{frame_filename}")
            print(f"Uploaded frame for Veo: {veo_gcs_uri}")
            
            # Upload Video to GCS for Gemini (better context)
            video_gcs_uri = upload_to_gcs(local_image_path, f"uploads/{filename}")
            gemini_part = Part.from_uri(video_gcs_uri, mime_type="video/mp4")
            prompt_context = "cette vidéo"
            
        else:
            # --- IMAGE HANDLING ---
            print(f"Detected image file: {filename}")
            
            # Upload Image to GCS for Veo
            veo_gcs_uri = upload_to_gcs(local_image_path, f"uploads/{filename}")
            
            # Prepare for Gemini (Pillow for safety)
            from PIL import Image as PILImage
            import io
            
            with PILImage.open(local_image_path) as img:
                print(f"Image opened. Mode: {img.mode}, Size: {img.size}")
                img = img.convert('RGB')
                buffer = io.BytesIO()
                img.save(buffer, format="JPEG", quality=85)
                image_data = buffer.getvalue()
            
            gemini_part = Part.from_data(image_data, mime_type="image/jpeg")
            prompt_context = "cette image"

        # Call Gemini
        model = GenerativeModel(GEMINI_MODEL_ID)
        prompt = f"""Vous êtes un expert en continuité visuelle. Décrivez {prompt_context} ou la dernière scène de manière ultra-détaillée. Concentrez-vous sur le personnage principal (apparence, vêtements), son émotion, l'environnement exact, l'heure de la journée et le style visuel (cinématique, éclairage, etc.). Votre description servira de base pour générer le clip suivant. Ne donnez que la description, pas de phrases d'introduction."""
        
        response = model.generate_content([gemini_part, prompt])
        description = response.text
        
        return {
            "gcs_uri": veo_gcs_uri, # Always return the IMAGE uri for Veo
            "description": description
        }
        
    except Exception as e:
        print(f"Error in Gemini/Frame analysis: {e}")
import os
import time
import json
import requests
from flask import Flask, render_template, request, jsonify
from google.cloud import aiplatform
from google.cloud import storage
import vertexai
from vertexai.preview.generative_models import GenerativeModel, Part, Image
import google.auth

app = Flask(__name__)


# Initialize Vertex AI
try:
    vertexai.init(project=PROJECT_ID, location=REGION)
except Exception as e:
    print(f"Warning: Vertex AI init failed (expected if no creds): {e}")

# --- Helper Functions ---

def upload_to_gcs(source_file_path, destination_blob_name):
    """Uploads a file to the bucket."""
    try:
        storage_client = storage.Client(project=PROJECT_ID)
        bucket = storage_client.bucket(BUCKET_NAME)
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(source_file_path)
        return f"gs://{BUCKET_NAME}/{destination_blob_name}"
    except Exception as e:
        print(f"Error uploading to GCS: {e}")
        raise

def analyze_last_frame_for_coherence(local_image_path):
    """
    1. Uploads the image/video to GCS.
    2. Uses Gemini to describe it.
    3. Returns the GCS URI of the *image* (frame) for Veo.
    """
    import cv2
    
    filename = os.path.basename(local_image_path)
    ext = os.path.splitext(filename)[1].lower()
    
    veo_gcs_uri = None
    gemini_part = None
    
    # Fallback to gemini-2.0-flash-exp
    GEMINI_MODEL_ID = "gemini-2.0-flash-exp" 

    try:
        if ext in ['.mp4', '.mov', '.avi', '.mkv']:
            # --- VIDEO HANDLING ---
            print(f"Detected video file: {filename}")
            
            # 1. Extract Last Frame for Veo
            cap = cv2.VideoCapture(local_image_path)
            if not cap.isOpened():
                print("Error: Could not open video.")
                raise ValueError("Could not open video file.")
            
            # Seek to end
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
            ret, frame = cap.read()
            cap.release()
            
            if not ret:
                print("Error: Could not read last frame.")
                raise ValueError("Could not extract last frame from video.")
            
            # Save frame locally
            frame_filename = f"frame_{filename}.jpg"
            frame_path = os.path.join("uploads", frame_filename)
            cv2.imwrite(frame_path, frame)
            print(f"Extracted last frame to {frame_path}")
            
            # Upload Frame to GCS for Veo
            veo_gcs_uri = upload_to_gcs(frame_path, f"uploads/{frame_filename}")
            print(f"Uploaded frame for Veo: {veo_gcs_uri}")
            
            # Upload Video to GCS for Gemini (better context)
            video_gcs_uri = upload_to_gcs(local_image_path, f"uploads/{filename}")
            gemini_part = Part.from_uri(video_gcs_uri, mime_type="video/mp4")
            prompt_context = "cette vidéo"
            
        else:
            # --- IMAGE HANDLING ---
            print(f"Detected image file: {filename}")
            
            # Upload Image to GCS for Veo
            veo_gcs_uri = upload_to_gcs(local_image_path, f"uploads/{filename}")
            
            # Prepare for Gemini (Pillow for safety)
            from PIL import Image as PILImage
            import io
            
            with PILImage.open(local_image_path) as img:
                print(f"Image opened. Mode: {img.mode}, Size: {img.size}")
                img = img.convert('RGB')
                buffer = io.BytesIO()
                img.save(buffer, format="JPEG", quality=85)
                image_data = buffer.getvalue()
            
            gemini_part = Part.from_data(image_data, mime_type="image/jpeg")
            prompt_context = "cette image"

        # Call Gemini
        model = GenerativeModel(GEMINI_MODEL_ID)
        prompt = f"""Vous êtes un expert en continuité visuelle. Décrivez {prompt_context} ou la dernière scène de manière ultra-détaillée. Concentrez-vous sur le personnage principal (apparence, vêtements), son émotion, l'environnement exact, l'heure de la journée et le style visuel (cinématique, éclairage, etc.). Votre description servira de base pour générer le clip suivant. Ne donnez que la description, pas de phrases d'introduction."""
        
        response = model.generate_content([gemini_part, prompt])
        description = response.text
        
        return {
            "gcs_uri": veo_gcs_uri, # Always return the IMAGE uri for Veo
            "description": description
        }
        
    except Exception as e:
        print(f"Error in Gemini/Frame analysis: {e}")
        # Fallback
        return {
            "gcs_uri": veo_gcs_uri if veo_gcs_uri else "gs://fallback/image.jpg",
            "description": "Description simulée (Erreur): Scène cinématique..."
        }

def generate_signed_url(gcs_uri):
    """Generates a v4 signed URL for a GCS blob."""
    try:
        storage_client = storage.Client(project=PROJECT_ID)
        
        # Parse gs://bucket/blob_name
        if not gcs_uri.startswith("gs://"):
            return gcs_uri # Already http?
            
        parts = gcs_uri[5:].split('/', 1)
        if len(parts) != 2:
            print(f"Invalid GCS URI: {gcs_uri}")
            return None
            
        bucket_name, blob_name = parts
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        
        url = blob.generate_signed_url(
            version="v4",
            expiration=3600, # 1 hour
            method="GET"
        )
        return url
    except Exception as e:
        print(f"Error generating signed URL: {e}")
        return None

def generate_next_clip_with_coherence(user_prompt, reference_image_uri, gemini_description):
    """
    1. Combines prompts.
    2. Calls Veo to generate video.
    """
    # 1. Construct Final Prompt
    final_prompt = f"{gemini_description}. Action: {user_prompt}"
    
    # 2. Call Veo (Video Generation)
    # Model: veo-3.1-generate-001
    # Using REST API with API Key and fetchPredictOperation
    
    try:
        # Veo is currently only available in us-central1
        VEO_REGION = "us-central1"
        MODEL_ID = "veo-3.1-generate-001"
        
        # Endpoint for Long Running Operation
        base_url = f"https://{VEO_REGION}-aiplatform.googleapis.com/v1"
        predict_url = f"{base_url}/projects/{PROJECT_ID}/locations/{VEO_REGION}/publishers/google/models/{MODEL_ID}:predictLongRunning"
        
        headers = {
            "Content-Type": "application/json; charset=utf-8",
            "X-Goog-Api-Key": VEO_API_KEY
        }
        
        # Determine mime type for the reference image
        # Veo requires explicit mimeType in the request
        # Note: analyze_last_frame_for_coherence now ensures we always get an image URI (even for videos)
        ref_ext = os.path.splitext(reference_image_uri)[1].lower()
        mime_type = "image/jpeg" # Default
        if ref_ext == '.png':
            mime_type = "image/png"
        
        # If it was a video, we extracted a frame which is .jpg, so it falls to default.
        print(f"DEBUG: Veo Input Image: {reference_image_uri} (Mime: {mime_type})")
        
        request_body = {
            "instances": [
                {
                    "prompt": final_prompt,
                    "image": {
                        "gcsUri": reference_image_uri,
                        "mimeType": mime_type
                    }
                }
            ],
            "parameters": {
                "aspectRatio": "16:9",
                "sampleCount": 1,
                "durationSeconds": 8,
                "personGeneration": "allow_all",
                "addWatermark": True,
                "includeRaiReason": True,
                "generateAudio": True,
                "resolution": "720p"
            }
        }
        
        print(f"DEBUG: Starting Long Running Operation at {predict_url}")
        
        # 1. Initiate LRO
        response = requests.post(predict_url, headers=headers, json=request_body)
        
        if response.status_code != 200:
            print(f"Veo LRO Init Error {response.status_code}: {response.text}")
            raise Exception(f"Veo Init Failed: {response.text}")
            
        operation = response.json()
        operation_name = operation.get('name')
        print(f"DEBUG: Operation started: {operation_name}")
        
        if not operation_name:
            print("Error: No operation name returned.")
            raise Exception("No operation name returned from Veo.")

        # 2. Poll for completion using fetchPredictOperation
        # As per Google's specific Veo documentation/snippet
        
        fetch_url = f"{base_url}/projects/{PROJECT_ID}/locations/{VEO_REGION}/publishers/google/models/{MODEL_ID}:fetchPredictOperation"
        
        print(f"DEBUG: Polling via fetchPredictOperation: {fetch_url}")
        
        for i in range(60):
            print(f"DEBUG: Polling operation... ({i+1}/60)")
            time.sleep(5)
            
            # The body must contain the full operation name
            fetch_body = {
                "operationName": operation_name
            }
            
            poll_response = requests.post(fetch_url, headers=headers, json=fetch_body)
            
            if poll_response.status_code != 200:
                print(f"Polling Error {poll_response.status_code}: {poll_response.text}")
                continue
                
            op_status = poll_response.json()
            
            # Check if done
            if op_status.get('done'):
                print("DEBUG: Operation done.")
                
                if 'error' in op_status:
                    print(f"Operation failed with error: {op_status['error']}")
                    raise Exception(f"Veo Operation Failed: {op_status['error']}")
                
                # Extract result
                if 'response' in op_status:
                    # The response field contains the PredictResponse
                    # It might contain 'videos' list
                    videos = op_status['response'].get('videos', [])
                    if videos:
                        first_video = videos[0]
                        
                        # Case 1: GCS URI (if configured to write to bucket)
                        if 'gcsUri' in first_video:
                            video_uri = first_video['gcsUri']
                            print(f"DEBUG: Generated Video URI: {video_uri}")
                            signed_url = generate_signed_url(video_uri)
                            if signed_url:
                                return signed_url
                            return video_uri
                            
                        # Case 2: Base64 Encoded Bytes (default if no bucket specified)
                        if 'bytesBase64Encoded' in first_video:
                            import base64
                            print("DEBUG: Received Base64 encoded video.")
                            
                            video_bytes = base64.b64decode(first_video['bytesBase64Encoded'])
                            
                            # Save locally
                            timestamp = int(time.time())
                            filename = f"generated_veo_{timestamp}.mp4"
                            local_path = os.path.join("uploads", filename)
                            
                            with open(local_path, "wb") as f:
                                f.write(video_bytes)
                            print(f"Saved generated video to {local_path}")
                            
                            # Upload to GCS to keep consistent flow (and get signed URL)
                            gcs_uri = upload_to_gcs(local_path, f"generated/{filename}")
                            print(f"Uploaded generated video to {gcs_uri}")
                            
                            signed_url = generate_signed_url(gcs_uri)
                            if signed_url:
                                return signed_url
                            return gcs_uri

                print(f"Unexpected LRO result format: {op_status}")
                raise Exception(f"Unexpected Veo result format: {op_status}")
        
        print("Timeout waiting for video generation.")
        raise Exception("Timeout waiting for Veo generation.")

    except Exception as e:
        print(f"Veo Generation Error (REST): {e}")
        # Re-raise to show error in frontend instead of fake video
        raise e

# --- Routes ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/jump_to', methods=['POST'])
def jump_to():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    user_prompt = request.form.get('prompt', '')
    
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    # Save locally first
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    
    local_path = os.path.join('uploads', file.filename)
    file.save(local_path)
    
    try:
        # 1. Analyze
        analysis_result = analyze_last_frame_for_coherence(local_path)
        
        # 2. Generate
        video_uri = generate_next_clip_with_coherence(
            user_prompt, 
            analysis_result['gcs_uri'], 
            analysis_result['description']
        )
        
        return jsonify({
            "gemini_description": analysis_result['description'],
            "video_uri": video_uri,
            "last_frame_uri": analysis_result['gcs_uri']
        })
    except Exception as e:
        print(f"Handler Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Ensure static folder exists for serving content if needed
    if not os.path.exists('static'):
        os.makedirs('static')
    app.run(debug=True, port=8080)
