# test_imports.py
import sys

print("--- Starting Import Test ---")
print(f"Using Python version: {sys.version}")

try:
    print("Testing: import streamlit...")
    import streamlit
    print("SUCCESS: Streamlit imported correctly.")

    print("\nTesting: import cv2...")
    import cv2
    print("SUCCESS: OpenCV imported correctly.")

    print("\nTesting: import PIL...")
    from PIL import Image
    print("SUCCESS: Pillow (PIL) imported correctly.")
    
    print("\nTesting: import face_recognition...")
    import face_recognition
    print("SUCCESS: face_recognition imported correctly.")

    print("\nTesting: from ultralytics import YOLO...")
    from ultralytics import YOLO
    print("SUCCESS: ultralytics (YOLO) imported correctly.")

    print("\n--- All major libraries imported successfully! ---")

except Exception as e:
    print(f"\n--- An error occurred ---")
    print(f"ERROR: {e}")