from google.cloud import aiplatform
import os

PROJECT_ID = os.environ.get("PROJECT_ID", "your_project_id")
REGION = os.environ.get("REGION", "us-central1")

aiplatform.init(project=PROJECT_ID, location=REGION)

print(f"Listing models in {PROJECT_ID} / {REGION}...")

try:
    # This lists custom models, but we want publisher models.
    # Publisher models are not always listable via this simple call, 
    # but we can try to check if we can instantiate the model the user wants
    # or list from the Model Garden if possible.
    
    # Actually, for GenerativeModel, we can try to list via the low-level API if possible.
    # But simpler: let's try to list all models and see if we find it.
    
    # Using the Model Garden API (PublisherModel)
    from google.cloud.aiplatform_v1 import ModelGardenServiceClient
    
    client = ModelGardenServiceClient(client_options={"api_endpoint": f"{REGION}-aiplatform.googleapis.com"})
    parent = f"publishers/google"
    
    # This might fail if permissions are strict, but let's try.
    # We want to list models from publisher 'google'.
    # The list_publisher_models method is available in v1.
    
    request = {"parent": parent}
    # Note: list_publisher_models might not be in the python library version installed?
    # Let's check dir() in the previous step... 
    # Actually, let's just try to instantiate a few known ones and the requested one and print errors.
    
    from vertexai.preview.generative_models import GenerativeModel
    
    candidates = [
        "gemini-3-pro-preview",
        "gemini-2.5-flash"
    ]
    
    print("\nTesting GenerativeModel instantiation:")
    for model_id in candidates:
        try:
            model = GenerativeModel(model_id)
            # Just creating the object doesn't validate existence in some SDK versions, 
            # we might need to generate to trigger the 404.
            print(f"Model object created for {model_id}, trying generation...")
            response = model.generate_content("Hello")
            print(f"SUCCESS: {model_id} exists and works.")
            break
        except Exception as e:
            print(f"FAILED: {model_id} - {e}")

except Exception as e:
    print(f"General Error: {e}")
