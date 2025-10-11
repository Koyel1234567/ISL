import os
import io
import base64
import requests
import cv2
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image
import time

# --- Flask App Setup ---
app = Flask(__name__) 
# Enable CORS to allow the HTML file to call this server from a different origin
CORS(app) 

# --- Configuration ---
# Get API Key from environment variables
API_KEY = os.environ.get("API_KEY", "")

# API Endpoints (Using the Gemini family of models for compatibility)
GEMINI_TRANSLATE_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={API_KEY}"
GEMINI_IMAGE_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-image-preview:generateContent?key={API_KEY}"

# Directory to save generated videos
VIDEO_DIR = 'videos'
if not os.path.exists(VIDEO_DIR):
    os.makedirs(VIDEO_DIR)

# --- Utility Functions ---

def fetch_gemini_api(url, payload, max_retries=3):
    """Handles API calls with exponential backoff and error checking."""
    headers = {'Content-Type': 'application/json'}
    
    for attempt in range(max_retries):
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=45)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            if response.status_code == 429 and attempt < max_retries - 1:
                print(f"Rate limit hit. Retrying in {2**(attempt+1)}s...")
                time.sleep(2**(attempt+1))
                continue
            print(f"API HTTP Error ({response.status_code}): {response.text}")
            raise RuntimeError(f"API Error ({response.status_code}): {response.text}")
        except requests.exceptions.RequestException as e:
            print(f"Request Error on attempt {attempt}: {e}")
            if attempt < max_retries - 1:
                time.sleep(2**(attempt+1))
                continue
            raise RuntimeError(f"Connection/Timeout Error: {e}")
    raise RuntimeError("API failed after multiple retries.")


def translate_to_gloss(english_text):
    """Translates English to ISL Gloss using the Gemini API."""
    system_prompt = "You are an expert International Sign Language (ISL) linguist. Translate the English sentence into a correct and grammatically sound ISL Gloss sequence. Output ONLY the ISL gloss, using ALL CAPS for signs and necessary Non-Manual Markers (NMMs) enclosed in brackets (e.g., [QUESTION], [HEAD-NOD])."

    payload = {
        "contents": [{"parts": [{"text": english_text}]}],
        "systemInstruction": {"parts": [{"text": system_prompt}]},
    }
    
    result = fetch_gemini_api(GEMINI_TRANSLATE_URL, payload)
    
    text = result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text')
    if not text:
        raise ValueError("Translation API returned empty text.")
    
    return text.strip()


def generate_frame_image(sign_word):
    """Generates a single image frame for a given sign word."""
    
    # Use a descriptive prompt for the AI
    if "[" in sign_word:
         # Handle Non-Manual Markers (NMMs)
        prompt_text = f"A simple, neutral AI avatar showing the expression related to: {sign_word.strip('[]')}. Close-up, dark background, photorealistic."
    else:
        # Create a detailed prompt for the signing action
        prompt_text = f"A photorealistic, kind AI avatar clearly performing the ISL sign for '{sign_word}', centered, close-up, black background, minimal motion blur, 4k."

    print(f"Generating image for sign: {sign_word}")

    # Payload structured for gemini-2.5-flash-image-preview model
    payload = {
        "contents": [{
            "parts": [{"text": prompt_text}]
        }],
        "generationConfig": {
            "responseModalities": ['IMAGE']
        },
    }

    result = fetch_gemini_api(GEMINI_IMAGE_URL, payload)

    # Extract base64 image data
    base64_data = result.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('inlineData', {}).get('data')

    if not base64_data:
        print(f"Error: No image data returned for sign {sign_word}.")
        return None

    # Convert base64 string to image bytes and then to OpenCV format (numpy array)
    image_bytes = base64.b64decode(base64_data)
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    return img_bgr

# --- API Endpoint ---

@app.route('/generate_video', methods=['POST'])
def generate_video_endpoint():
    """
    Main endpoint: Translates English -> Generates Frames -> Stitches Video.
    """
    try:
        data = request.get_json()
        english_text = data.get('english_text')
        
        if not english_text:
            return jsonify({"error": "Missing 'english_text' in request body"}), 400

        # 1. Translate to ISL Gloss
        print(f"\nProcessing English: '{english_text}'")
        isl_gloss = translate_to_gloss(english_text)
        print(f"Translated Gloss: {isl_gloss}")

        # 2. Tokenization: Split the gloss into individual signs/tokens
        signs = [token for token in isl_gloss.upper().split() if token]
        
        frames = []
        
        # 3. Frame Generation (Sequential API Calls)
        for sign in signs:
            frame = generate_frame_image(sign)
            if frame is not None:
                frames.append(frame)
            else:
                print(f"Skipping sign due to frame generation failure: {sign}")

        if not frames:
            return jsonify({"error": "Failed to generate any video frames. Check your API key access."}), 500
        
        # --- 4. Stitching & Encoding (using OpenCV) ---
        
        frame_height, frame_width, _ = frames[0].shape
        # Adjust frame rate and repetition to make signs readable
        FRAME_RATE = 10.0  # Frames per second
        FRAME_REPETITION = 30 # Repeat each sign for 30 frames (3 seconds per sign)

        video_filename = f"isl_output_{os.urandom(4).hex()}.mp4"
        video_path = os.path.join(VIDEO_DIR, video_filename)

        # Use 'mp4v' codec for broader compatibility with different OpenCV backends
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        out = cv2.VideoWriter(video_path, fourcc, FRAME_RATE, (frame_width, frame_height))

        for frame in frames:
            # Repeat the frame to slow down the perceived signing speed
            for _ in range(FRAME_REPETITION):
                out.write(frame)
        
        out.release()
        print(f"Successfully generated video: {video_path}")

        # --- 5. Return Video URL ---
        # The URL points back to the Flask server to serve the video file.
        video_url = f"http://localhost:5000/{VIDEO_DIR}/{video_filename}"
        return jsonify({"video_url": video_url, "isl_gloss": isl_gloss})

    except Exception as e:
        print(f"Fatal server error: {e}")
        return jsonify({"error": str(e)}), 500

# --- Serve Static Files (The Generated Video) ---

@app.route(f'/{VIDEO_DIR}/<path:filename>')
def serve_video(filename):
    """
    Allows the browser to fetch the generated MP4 file.
    """
    return send_from_directory(VIDEO_DIR, filename)

if __name__ == '__main__':
    # Make sure we are running in debug mode for development
    if not API_KEY:
         print("WARNING: API_KEY not set. Check environment variables.")
    app.run(debug=True, port=5000)
