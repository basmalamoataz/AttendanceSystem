# ===============================
# Libraries we need
# ===============================
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
import base64

# --- Helper function to load and encode images ---
def image_to_base64(img_path):  
    try:
        with open(img_path, "rb") as f:
            return base64.b64encode(f.read()).decode()  
        """I used base64 because
          Streamlit and HTML often embed images in base64 when they can’t load from file paths directly."""
    except FileNotFoundError:
        st.error(f"Image not found at {img_path}")
        return None

# ===============================
# YOLO and "known_faces" Loading Functions
# ===============================

# Use st.cache_resource to load the YOLO model only once
@st.cache_resource
def load_yolo_model():
    """Loads the YOLO model and caches it."""
    # We import YOLO here, inside the function, to avoid startup conflicts.
    from ultralytics import YOLO

    model_path = "models/best (1).pt"
    if os.path.exists(model_path):
        model = YOLO(model_path)
        return model
    else:
        st.error(f"Fatal Error: YOLO model not found at path: {model_path}")
        st.stop() # Stop the app if the model is not found

# Use st.cache_data to load and process known faces only once
@st.cache_data
def load_known_faces():
    """Loads and encodes all known faces from the directory."""
    KNOWN_FACES_DIR = "known_faces"
    known_faces_encodings = [] # List to hold encodings
    known_faces_names = [] # List to hold corresponding names

    if not os.path.exists(KNOWN_FACES_DIR):
        st.error(f"Fatal Error: The '{KNOWN_FACES_DIR}' directory was not found.")
        st.info("Please create this folder and add sub-folders for each known person.")
        st.stop()

    for person in os.listdir(KNOWN_FACES_DIR): # it loops through each folder in the known_faces directory
        person_dir = os.path.join(KNOWN_FACES_DIR, person)  # person_dir: The full path to the folder for one person's images.
        if not os.path.isdir(person_dir):  
            continue

        for img_name in os.listdir(person_dir):  # Loops through every image inside each person's folder.
            img_path = os.path.join(person_dir, img_name) #
            try:
                # We need to import face_recognition here as well for caching
                import face_recognition
                img = face_recognition.load_image_file(img_path) # Load the image file
                encodings = face_recognition.face_encodings(img) # Get the face encodings
                if encodings:
                    known_faces_encodings.append(encodings[0]) # We take the first encoding found in the image
                    known_faces_names.append(person) # Associate the encoding with the person's name
            except Exception as e:
                st.warning(f"Could not process image {img_path}: {e}")
    
    if not known_faces_encodings:
        st.warning("Warning: No known faces were loaded. Please check the 'known_faces' directory.")
        
    return known_faces_encodings, known_faces_names


# ===============================
# Main Application Logic
# ===============================

def main():
    st.set_page_config(page_title="Attendance System", layout="wide")

    # --- Display Header ---
    eye_logo_b64 = image_to_base64("Gemini_Generated_Image_pit6rspit6rspit6-removebg-preview.png")
    nti_logo_b64 = image_to_base64("logo.png")
    
    if eye_logo_b64 and nti_logo_b64:
        st.markdown(
            f"""
            <style>
            .header-row {{ display: grid; grid-template-columns: 1fr auto 1fr; align-items: center; gap: 25px; width: 90%; margin: auto; margin-bottom: 40px; }}
            .eye-logo {{ height: 300px; justify-self: end; }}
            .nti-logo {{ height: 100px; justify-self: start; }}
            .header-title {{ font-size: 4rem; font-weight: bold; color: white; white-space: nowrap; transform: translateX(-40px); }}
            </style>
            <div class="header-row">
                <img src="data:image/png;base64,{eye_logo_b64}" class="eye-logo" />
                <span class="header-title">Attendance System</span>
                <img src="data:image/png;base64,{nti_logo_b64}" class="nti-logo" />
            </div>
            """,
            unsafe_allow_html=True
        )

    # --- Load models and faces with a spinner ---
    with st.spinner('Loading models and encoding faces, please wait...'):
        detector = load_yolo_model()
        known_faces, known_names = load_known_faces()


    # --- Sidebar for user input ---
    st.sidebar.title("Upload Options")
    option = st.sidebar.radio("Choose input source:", ["Image", "Video", "Webcam"])


    # --- Face Recognition Function (uses loaded models) ---
    def recognize_faces_in_frame(frame):
        import face_recognition 
        attendees = [] # List to hold names of detected attendees
        results = detector(frame, verbose=False) # Runs the face detector model on the input frame. and "verbose=False" prevents detailed output
        boxes = results[0].boxes.xyxy.cpu().numpy()  # Extract bounding boxes from the detection results

        for box in boxes:
            x1, y1, x2, y2 = [int(v) for v in box[:4]] # Convert coordinates to integers (x1, y1) is the top-left corner, (x2, y2) is the bottom-right corner 
            face = frame[y1:y2, x1:x2] # Crop the face from the frame using the bounding box coordinates
            if face.size == 0: continue

            rgb_face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB) # Convert the cropped face to RGB format to be compatible with face_recognition
            encodings = face_recognition.face_encodings(rgb_face) # Get the face encodings for the cropped face

            name = "Unknown" # Default name
            if encodings:
                emb = encodings[0] # We take the first encoding found
                matches = face_recognition.compare_faces(known_faces, emb, tolerance=0.5) # Compare the encoding with known faces
                face_distances = face_recognition.face_distance(known_faces, emb) # Get distances to known faces and the smallest distance indicates the best match
                if True in matches:
                    best_match_index = np.argmin(face_distances) # Get the index of the best match
                    if matches[best_match_index]:
                        name = known_names[best_match_index] # Get the name of the best match
            
            attendees.append(name) # Add the detected name to the attendees list
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255) # Green for known, Red for unknown
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2) # Draw bounding box
            cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2) # Put the name label above the box

        return frame, list(set(attendees))


    # --- Handle Image Upload ---
    if option == "Image":
        uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if uploaded_image:
            img = Image.open(uploaded_image).convert('RGB') # Convert to RGB because OpenCV uses BGR by default
            img_array = np.array(img)
            frame_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR) # Convert RGB to BGR for OpenCV
            processed_frame, attendees = recognize_faces_in_frame(frame_bgr) # Process the image
            st.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB), caption="Processed Image") # Convert back to RGB for displaying in Streamlit
            st.success(f"✅ Attendees detected: {', '.join(attendees)}")

    # --- Handle Video Upload ---
    elif option == "Video":
      uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
      if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        cap = cv2.VideoCapture(tfile.name)

        # Placeholders for the video frames and the status text
        stframe = st.empty()
        status_text = st.empty() 

        # This set will store the names of people who have been positively identified.
        # Once a name is in here, it stays.
        confirmed_attendees = set()

        status_text.info("Processing video, please wait...")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Recognize faces in the current frame
            frame, current_attendees = recognize_faces_in_frame(frame)

            
            # Iterate through the people found in THIS frame
            for name in current_attendees:
                # If the person is a known individual, add them to our master list
                if name != "Unknown":
                    confirmed_attendees.add(name)

            # Display the processed video frame
            stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            # Display the running list of confirmed attendees below the video
            if confirmed_attendees:
                 status_text.success(f"Confirmed Attendees: {', '.join(sorted(list(confirmed_attendees)))}")
            else:
                 status_text.info("Processing... No known attendees confirmed yet.")


        cap.release()

        # Display the final results
        if confirmed_attendees:
            st.success(f"✅ Final Attendees Detected: {', '.join(sorted(list(confirmed_attendees)))}")
        else:
            st.warning("⏹️ Processing complete. No known attendees were identified in the video.")

    # --- Handle Webcam ---
    elif option == "Webcam":
        st.info("Allow camera access and click 'Take a photo' to capture an image.")
        camera = st.camera_input("Take a photo")
        if camera:
            img = Image.open(camera).convert('RGB')
            img_array = np.array(img)
            frame_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            processed_frame, attendees = recognize_faces_in_frame(frame_bgr)
            st.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB), caption="Processed Webcam Frame")
            st.success(f"✅ Attendees detected: {', '.join(attendees)}")


if __name__ == '__main__':
    main()

