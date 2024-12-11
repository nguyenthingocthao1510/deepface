import os
import json
import cv2
import numpy as np
import faiss
from deepface import DeepFace

hr_database_folder = r'C:\Users\20521\OneDrive\Desktop\nckh\deepface\hr_database'
capture_image_folder = 'captured'
metadata_folder = 'metadata'
outcome_folder = 'outcome'
processing_image_folder = 'processing_image'

os.makedirs(capture_image_folder, exist_ok=True)
os.makedirs(metadata_folder, exist_ok=True)
os.makedirs(outcome_folder, exist_ok=True)
os.makedirs(processing_image_folder, exist_ok=True)

def load_hr_database():
    hr_database_files = []
    for folder_name in os.listdir(hr_database_folder):
        folder_path = os.path.join(hr_database_folder, folder_name)
        if os.path.isdir(folder_path):
            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)
                if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    hr_database_files.append({'file': file_path, 'name': folder_name})
    return hr_database_files



def capture_images():
    print("Opening camera for image capture...")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Camera could not be opened.")
        return

    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame. Exiting...")
            break

        cv2.imshow('Capture Face (Press SPACE to Capture, Q to Quit)', frame)
        key = cv2.waitKey(1)
        if key == ord(' '):
            capture_path = os.path.join(capture_image_folder, f"captured_{count}.jpg")
            cv2.imwrite(capture_path, frame)
            print(f"Image saved: {capture_path}")
            count += 1
        elif key == ord('q'):
            print("Exiting capture mode.")
            break

    cap.release()
    cv2.destroyAllWindows()

def process_captured_images():
    print("Processing captured images with normalization...")
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    for file in os.listdir(capture_image_folder):
        image_path = os.path.join(capture_image_folder, file)
        output_path = os.path.join(processing_image_folder, file)

        try:
            image = cv2.imread(image_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            if len(faces) > 0:
                x, y, w, h = faces[0]
                cropped_face = image[y:y+h, x:x+w]
                cropped_gray = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2GRAY)
                equalized_face = cv2.equalizeHist(cropped_gray)
                normalized_face = cv2.cvtColor(equalized_face, cv2.COLOR_GRAY2BGR)
                cv2.imwrite(output_path, normalized_face)
                print(f"Processed and saved (normalized): {output_path}")
            else:
                print(f"No face detected in {file}.")
        except Exception as e:
            print(f"Error processing {file}: {e}")

def build_faiss_index(hr_database_files):
    hr_embeddings = []
    hr_names = []

    for hr_file in hr_database_files:
        try:
            hr_representation = DeepFace.represent(
                img_path=hr_file['file'],
                model_name='Facenet512',
                detector_backend='retinaface'
            )
            if hr_representation:
                hr_embeddings.append(hr_representation[0]["embedding"])
                hr_names.append(hr_file['name'])
        except Exception as e:
            print(f"Error processing {hr_file['file']}: {e}")

    hr_embeddings = np.array(hr_embeddings).astype('float32')
    dimension = hr_embeddings.shape[1]

    faiss.normalize_L2(hr_embeddings)
    index = faiss.IndexFlatIP(dimension)
    index.add(hr_embeddings)

    return index, hr_names

def search_with_faiss(query_embedding, index, hr_names, k):
    query_embedding = np.array([query_embedding]).astype('float32')
    faiss.normalize_L2(query_embedding)
    distances, indices = index.search(query_embedding, k)
    results = [(hr_names[i], distances[0][idx]) for idx, i in enumerate(indices[0])]
    return results

def search_in_hr_database(index, hr_names):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    for file in os.listdir(capture_image_folder):
        image_path = os.path.join(capture_image_folder, file)
        try:
            representation = DeepFace.represent(
                img_path=image_path,
                model_name='Facenet512',
                detector_backend='retinaface'
            )
            if representation:
                query_embedding = representation[0]["embedding"]
                results = search_with_faiss(query_embedding, index, hr_names, k=len(hr_names))
                top_match_name, top_similarity = results[0]

                if top_similarity > 0.4:
                    print(f"name: {top_match_name} - similarity: ({top_similarity:.2f})")
                else:
                    print(f"Unknown - similarity: ({top_similarity:.2f})")
                
                image = cv2.imread(image_path)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

                for (x, y, w, h) in faces:
                    label = f"{top_match_name} ({top_similarity:.2f})"
                    cv2.rectangle(image, (x, y), (x + w, y + h), (204, 255, 229), 2)
                    cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (204, 255, 229), 2)

                outcome_path = os.path.join(outcome_folder, file)
                cv2.imwrite(outcome_path, image)
                print(f"Outcome saved: {outcome_path}")
        except Exception as e:
            print(f"Error searching for {file}: {e}")

def main():
    global hr_database_files
    hr_database_files = load_hr_database()
    capture_images()
    process_captured_images()
    index, hr_names = build_faiss_index(hr_database_files)
    search_in_hr_database(index, hr_names)

if __name__ == "__main__":
    main()
