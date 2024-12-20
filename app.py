import os
import json
import cv2
import numpy as np
import faiss
from deepface import DeepFace
from retinaface import RetinaFace

# Define folders
hr_database_folder = 'hr_database'
capture_image_folder = 'captured'
metadata_folder = 'metadata'
outcome_folder = 'outcomes'
processing_image_folder = 'processing_image'

# Create folders if they don't exist
os.makedirs(capture_image_folder, exist_ok=True)
os.makedirs(metadata_folder, exist_ok=True)
os.makedirs(outcome_folder, exist_ok=True)
os.makedirs(processing_image_folder, exist_ok=True)

def capture_images():
    cap = cv2.VideoCapture(0)

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

# Process images with labels
def process_images_with_labels(input_folder, output_folder):
    processed_files = []
    for root, _, files in os.walk(input_folder):
        for file_name in files:
            image_path = os.path.join(root, file_name)
            try:
                print(f"Processing image: {image_path}")
                image = cv2.imread(image_path)
                if image is None:
                    print(f"Error: Unable to read {image_path}. Skipping...")
                    continue

                label = os.path.basename(root)
                faces = RetinaFace.extract_faces(img_path=image_path, align=True)
                if not faces:
                    print(f"No faces detected in {file_name}. Skipping...")
                    continue

                face_np = faces[0]
                face_np = cv2.medianBlur(face_np, 1)
                face_yuv = cv2.cvtColor(face_np, cv2.COLOR_BGR2YUV)
                face_yuv[:, :, 0] = cv2.equalizeHist(face_yuv[:, :, 0])
                face_np = cv2.cvtColor(face_yuv, cv2.COLOR_YUV2BGR)
                face_np = cv2.bilateralFilter(face_np, 0, 0, 0)

                output_path = os.path.join(output_folder, f"processed_{file_name}")
                cv2.imwrite(output_path, face_np)
                processed_files.append({"file": output_path, "label": label})
            except Exception as e:
                print(f"Error processing {file_name}: {e}")
    return processed_files

def save_embeddings_with_labels(image_files, metadata_file, model_name="GhostFaceNet"):
    embeddings = []
    for image_info in image_files:
        image_path = image_info["file"]
        label = image_info["label"]
        try:
            embedding_objs = DeepFace.represent(
                img_path=image_path,
                model_name=model_name,
                detector_backend='retinaface',
                enforce_detection=True
            )
            for embedding_obj in embedding_objs:
                embedding = embedding_obj["embedding"]
                embeddings.append({"file": image_path, "embedding": embedding, "label": label})
        except Exception as e:
            print(f"Error generating embedding for {image_path}: {e}")
    
    with open(metadata_file, 'w') as f:
        json.dump(embeddings, f)
    print(f"Saved embeddings to {metadata_file}")
    return embeddings

def build_faiss_index_from_metadata_with_labels(metadata_file):
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    if not metadata:
        raise ValueError("Metadata file is empty or no valid embeddings were found.")

    embeddings = np.array([item['embedding'] for item in metadata]).astype('float32')
    if embeddings.size == 0:
        raise ValueError("No embeddings found in metadata.")

    labels = [item['label'] for item in metadata]

    faiss.normalize_L2(embeddings)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    return index, labels

def search_in_hr_database(capture_metadata, index, hr_labels, threshold=0.3800):
    if not capture_metadata:
        print("Error: No metadata found for captured images.")
        return []

    outcomes = []
    for capture in capture_metadata:
        try:
            query_embedding = np.array([capture['embedding']]).astype('float32')
            faiss.normalize_L2(query_embedding)
            distances, indices = index.search(query_embedding, k=1)

            best_distance = distances[0][0]
            best_index = indices[0][0]

            if best_distance >= threshold and best_index < len(hr_labels):
                label = hr_labels[best_index]
                outcomes.append({
                    "label": label,
                    "distance": best_distance
                })
            else:
                outcomes.append({
                    "label": "unknown",
                    "distance": best_distance
                })
        except Exception as e:
            print(f"Error processing {capture['file']}: {e}")
    return outcomes

def save_outcome(image_file, label, best_distance):
    image_path = os.path.join(capture_image_folder, image_file)
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to read image {image_file}. Skipping...")
        return

    final_label = label
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (204, 255, 229), 2)
        text_label = f"{final_label} - {best_distance:.4f}"
        cv2.putText(image, text_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (204, 255, 229), 2)
    
    outcome_path = os.path.join(outcome_folder, os.path.basename(image_file))
    cv2.imwrite(outcome_path, image)
    print(f"Saved outcome to {outcome_path}")

def process_hr_database_images(hr_folder, output_folder, metadata_file):
    processed_files = process_images_with_labels(hr_folder, output_folder)
    if not processed_files:
        return []

    embeddings = save_embeddings_with_labels(processed_files, metadata_file)
    return embeddings

# Main function
def main():
    capture_images()

    captured_processed_files = process_images_with_labels(capture_image_folder, processing_image_folder)
    capture_metadata_file = os.path.join(metadata_folder, 'capture_metadata.json')
    capture_metadata = save_embeddings_with_labels(captured_processed_files, capture_metadata_file)

    hr_metadata_file = os.path.join(metadata_folder, 'hr_metadata.json')
    hr_embeddings = process_hr_database_images(hr_database_folder, processing_image_folder, hr_metadata_file)

    if not hr_embeddings:
        return

    index, hr_labels = build_faiss_index_from_metadata_with_labels(hr_metadata_file)

    outcomes = search_in_hr_database(capture_metadata, index, hr_labels)

    capture_files = [f for f in os.listdir(capture_image_folder) if os.path.isfile(os.path.join(capture_image_folder, f))]
    for capture_file, outcome in zip(capture_files, outcomes):
        save_outcome(image_file=capture_file, label=outcome["label"], best_distance=outcome["distance"])

if __name__ == "__main__":
    main()
