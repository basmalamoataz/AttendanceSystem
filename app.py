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
import requests
import zipfile

# --- Helper function to load and encode images ---
def image_to_base64(img_path):  
    try:
        with open(img_path, "rb") as f:
            return base64.b64encode(f.read()).decode()  
    except FileNotFoundError:
        st.error(f"Image not found at {img_path}")
        return None

# ===============================
# Download helpers
# ===============================

def download_yolo_model():
    """Download YOLO best.pt from Google Drive if missing"""
    model_path = "models/best.pt"
    if not os.path.exists(model_path):
        st.warning("⚠️ YOLO model not found locally. Downloading...")
        os.makedirs("models", exist_ok=True)

        # Replace with your Google Drive file ID
        file_id = "1JFaIXJ8IAJRLpZ3tS0RFjnsDNP39LFkX"
        download_url = f"https://drive.google.com/uc?export=download&id={file_id}"

        r = requests.get(download_url)
        if r.status_code == 200:
            with open(model_path, "wb") as f:
                f.write(r.content)
            st.success("✅ YOLO model downloaded successfully.")
        else:
            st.error("❌ Failed to download YOLO model from Google Drive")
            st.stop()
    return model_path


def download_and_extract_known_faces():
    """Download and extract known_faces.zip from Google Drive if missing"""
    folder_path = "known_faces"
    if not os.path.exists(folder_path):
        st.warning("⚠️ known_faces folder not found. Downloading...")
        file_id = "1_aWYXEIK3U0VLHHUS36zoWUY87jHr8hT"  # your Drive zip file
        download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
        zip_path = "known_faces.zip"

        r = requests.get(download_url)
        if r.status_code == 200:
            with open(zip_path, "wb") as f:
                f.write(r.content)

            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(".")
            os.remove(zip_path)
            st.success("✅ known_faces folder downloaded & extracted.")
        else:
            st.error("❌ Failed to download known_faces.zip from Google Drive")
            st.stop()

# ===============================
# YOLO and "known_faces" Loading Functions
# ===============================

@st.cache_resource
def load_yolo_model():
    from ultralytics import YOLO
    model_path = download_yolo_model()
    return YOLO(model_path)

@st.cache_data
def load_known_faces():
    import face_recognition
    KNOWN_FACES_DIR = "known_faces"
    known_faces_encodings = []
    known_faces_names = []

    if not os.path.exists(KNOWN_FACES_DIR):
        st.error(f"Fatal Error: '{KNOWN_FACES_DIR}' not found.")
        st.stop()

    for person in os.listdir(KNOWN_FACES_DIR):
        person_dir = os.path.join(KNOWN_FACES_DIR, person)
        if not os.path.isdir(person_dir):  
            continue

        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)
            try:
                img = face_recognition.load_image_file(img_path)
                encodings = face_recognition.face_encodings(img)
                if encodings:
                    known_faces_encodings.append(encodings[0])
                    known_faces_names.append(person)
            except Exception as e:
                st.warning(f"Could not process {img_path}: {e}")

    if not known_faces_encodings:
        st.warning("⚠️ No known faces were loaded.")
        
    return known_faces_encodings, known_faces_names

# ===============================
# Main Application Logic
# ===============================

def main():
    st.set_page_config(page_title="Attendance System", layout="wide")

    # ✅ Make sure data is ready
    download_and_extract_known_faces()

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

    # --- Sidebar ---
    st.sidebar.title("Upload Options")
    option = st.sidebar.radio("Choose input source:", ["Image", "Video", "Webcam"])

    # --- Face Recognition ---
    def recognize_faces_in_frame(frame):
        import face_recognition 
        attendees = []
        results = detector(frame, verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy()

        for box in boxes:
            x1, y1, x2, y2 = [int(v) for v in box[:4]]
            face = frame[y1:y2, x1:x2]
            if face.size == 0: continue

            rgb_face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            encodings = face_recognition.face_encodings(rgb_face)

            name = "Unknown"
            if encodings:
                emb = encodings[0]
                matches = face_recognition.compare_faces(known_faces, emb, tolerance=0.5)
                face_distances = face_recognition.face_distance(known_faces, emb)
                if True in matches:
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = known_names[best_match_index]
            
            attendees.append(name)
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return frame, list(set(attendees))

    # --- Handle Input Sources ---
    if option == "Image":
        uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if uploaded_image:
            img = Image.open(uploaded_image).convert('RGB')
            frame_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            processed_frame, attendees = recognize_faces_in_frame(frame_bgr)
            st.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB), caption="Processed Image")
            st.success(f"✅ Attendees: {', '.join(attendees)}")

    elif option == "Video":
        uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
        if uploaded_video:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_video.read())
            cap = cv2.VideoCapture(tfile.name)
            stframe, status_text = st.empty(), st.empty()
            confirmed_attendees = set()
            status_text.info("Processing video...")

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                frame, current_attendees = recognize_faces_in_frame(frame)
                for name in current_attendees:
                    if name != "Unknown":
                        confirmed_attendees.add(name)
                stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if confirmed_attendees:
                    status_text.success(f"Confirmed: {', '.join(sorted(confirmed_attendees))}")
                else:
                    status_text.info("Processing...")

            cap.release()
            if confirmed_attendees:
                st.success(f"✅ Final Attendees: {', '.join(sorted(confirmed_attendees)))}")
            else:
                st.warning("⏹️ No known attendees identified.")

    elif option == "Webcam":
        st.info("Allow camera access and click 'Take a photo'.")
        camera = st.camera_input("Take a photo")
        if camera:
            img = Image.open(camera).convert('RGB')
            frame_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            processed_frame, attendees = recognize_faces_in_frame(frame_bgr)
            st.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB), caption="Processed Webcam Frame")
            st.success(f"✅ Attendees: {', '.join(attendees)}")

if __name__ == '__main__':
    main()
