import os
import json
import cv2
import numpy as np
import faiss
from deepface import DeepFace
from retinaface import RetinaFace

# Cấu hình các thư mục
hr_database_folder = r'C:\Users\20521\OneDrive\Desktop\nckh\deepface\hr_database'
capture_image_folder = 'captured'
metadata_folder = 'metadata'
outcome_folder = 'outcome'
processing_image_folder = 'processing_image'

os.makedirs(capture_image_folder, exist_ok=True)
os.makedirs(metadata_folder, exist_ok=True)
os.makedirs(outcome_folder, exist_ok=True)
os.makedirs(processing_image_folder, exist_ok=True)

# Chức năng chụp ảnh
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

# Xử lý hình ảnh bằng RetinaFace
def process_images_with_labels(input_folder, output_folder):
    processed_files = []
    for root, dirs, files in os.walk(input_folder):  # Lặp qua toàn bộ thư mục
        for file_name in files:
            image_path = os.path.join(root, file_name)
            try:
                print(f"Processing image: {image_path}")
                image = cv2.imread(image_path)
                if image is None:
                    print(f"Error: Unable to read {image_path}. Skipping...")
                    continue

                # Phát hiện khuôn mặt
                faces = RetinaFace.extract_faces(img_path=image_path, align=True)
                if not faces:
                    print(f"No faces detected in {file_name}. Skipping...")
                    continue

                # Chỉ lấy khuôn mặt đầu tiên
                face_np = faces[0]
                face_np = cv2.medianBlur(face_np, 5)
                face_yuv = cv2.cvtColor(face_np, cv2.COLOR_BGR2YUV)
                face_yuv[:, :, 0] = cv2.equalizeHist(face_yuv[:, :, 0])
                face_np = cv2.cvtColor(face_yuv, cv2.COLOR_YUV2BGR)
                face_np = cv2.bilateralFilter(face_np, 9, 75, 75)

                # Lưu ảnh đã xử lý
                output_path = os.path.join(output_folder, f"processed_{file_name}")
                cv2.imwrite(output_path, face_np)
                processed_files.append({"file": output_path, "label": file_name.split('.')[0]})
            except Exception as e:
                print(f"Error processing {file_name}: {e}")
    return processed_files

# Lưu embeddings
def save_embeddings_with_labels(image_files, metadata_file, model_name="Human-beings"):
    embeddings = []
    for image_info in image_files:
        image_path = image_info["file"]
        label = image_info["label"]
        try:
            embedding_objs = DeepFace.represent(
                img_path=image_path,
                model_name=model_name,
                detector_backend='retinaface',
                enforce_detection=False
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

# Xây dựng FAISS index
def build_faiss_index_from_metadata_with_labels(metadata_file):
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    if not metadata:
        raise ValueError("Metadata file is empty or no valid embeddings were found.")

    embeddings = np.array([item['embedding'] for item in metadata]).astype('float32')
    if embeddings.size == 0:
        raise ValueError("No embeddings found in metadata.")

    labels = [item['label'] for item in metadata]

    # Chuẩn hóa embeddings trước khi thêm vào FAISS
    faiss.normalize_L2(embeddings)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    return index, labels

# Tìm kiếm trong cơ sở dữ liệu HR
def search_in_hr_database(capture_metadata, index, hr_labels, threshold=0.57):
    """
    Search for captured image embeddings in the HR database using FAISS.
    Print 'label - similarity' for matches or 'unknown - similarity' otherwise.
    """
    if not capture_metadata:
        print("Error: No metadata found for captured images.")
        return

    for capture in capture_metadata:
        try:
            query_embedding = np.array([capture['embedding']]).astype('float32')
            faiss.normalize_L2(query_embedding)
            distances, indices = index.search(query_embedding, k=1)

            print(f"\nSearch results for {capture['file']}:")
            best_distance = distances[0][0]  # Độ tương tự cao nhất
            best_index = indices[0][0]  # Index tương ứng

            if best_distance >= threshold and best_index < len(hr_labels):
                label = hr_labels[best_index]
                print(f"{label} - {best_distance:.4f}")
            else:
                print(f"unknown - {best_distance:.4f}")

        except Exception as e:
            print(f"Error processing {capture['file']}: {e}")

# Xử lý cơ sở dữ liệu HR
def process_hr_database_images(hr_folder, output_folder, metadata_file):
    print("Processing HR database images...")
    processed_files = process_images_with_labels(hr_folder, output_folder)
    if not processed_files:
        print("No valid processed files found in HR database.")
        return []

    print("Saving HR embeddings to metadata...")
    embeddings = save_embeddings_with_labels(processed_files, metadata_file)
    return embeddings

# Hàm chính
def main():
    print("Capturing and processing images...")
    capture_images()  

    print("Processing captured images...")
    captured_processed_files = process_images_with_labels(capture_image_folder, processing_image_folder)

    print("Saving captured embeddings...")
    capture_metadata_file = os.path.join(metadata_folder, 'capture_metadata.json')
    capture_metadata = save_embeddings_with_labels(captured_processed_files, capture_metadata_file)

    print("Processing HR database...")
    hr_metadata_file = os.path.join(metadata_folder, 'hr_metadata.json')
    hr_embeddings = process_hr_database_images(hr_database_folder, processing_image_folder, hr_metadata_file)

    if not hr_embeddings:
        print("No valid HR database metadata found. Exiting.")
        return

    print("Building FAISS index from HR database...")
    index, hr_labels = build_faiss_index_from_metadata_with_labels(hr_metadata_file)

    print("Searching HR database for captured images...")
    search_in_hr_database(capture_metadata, index, hr_labels)

if __name__ == "__main__":
    main()
