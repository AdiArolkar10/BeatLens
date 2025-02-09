import streamlit as st
import google.generativeai as genai
import torch
import numpy as np
from ultralytics import YOLO
from PIL import Image

# Configure Gemini API (Replace with your API key)
genai.configure(api_key="Your api key")

# Load YOLOv8 Model
model = YOLO("yolov8n.pt")

# Function to detect objects using YOLOv8
def detect_objects(image):
    image = np.array(image)  # Convert PIL Image to NumPy array
    results = model(image)  # Run YOLO on the image
    
    detected_objects = set()
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls)  # Get class index
            detected_objects.add(result.names[class_id])  # Get object name

    return list(detected_objects)

# Function to get song recommendations from Gemini API
def get_song_recommendations(objects, user_prompt):
    if not objects and not user_prompt:
        return ["No objects detected, please try another image."]

    prompt = f"Recommend 3 songs based on these detected objects: {', '.join(objects)}"
    if user_prompt:
        prompt += f" and consider this user preference: {user_prompt}"
    
    model = genai.GenerativeModel("gemini-pro")
    
    try:
        response = model.generate_content(prompt)
        songs = response.text.split("\n")[:3]  # Get first 3 lines as song suggestions
        return songs if songs else ["No song recommendations found."]
    except Exception as e:
        return [f"Error getting recommendations: {str(e)}"]

# Streamlit UI
st.title("🎵 BeatLens")
st.write("Upload an image, and we'll suggest songs based on the detected objects!")

uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

# User prompt input
user_prompt = st.text_input("🎤 Want specific song suggestions? Describe your mood or preference!")

if uploaded_image:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Detecting objects..."):
        objects = detect_objects(image)

    if objects:
        st.success(f"Detected Objects: {', '.join(objects)}")

        with st.spinner("Getting song recommendations..."):
            songs = get_song_recommendations(objects, user_prompt)

        st.subheader("🎶 Recommended Songs:")
        for song in songs:
            st.write(f"- {song}")
    else:
        st.warning("No objects detected. Try another image.")
