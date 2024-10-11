import os
import json
import cv2
from deepface import DeepFace
import time
from sklearn.metrics.pairwise import cosine_similarity

hr_database_files = [
    { 'file': 'C:/Users/20521/OneDrive/Desktop/test_facenet/MQ1.jpg', 'name': 'MQ.jpg' },
    { 'file': 'C:/Users/20521/OneDrive/Desktop/test_facenet/NH8.jpg', 'name': 'NH.jpg' },
    { 'file': 'C:/Users/20521/OneDrive/Desktop/test_facenet/QV1.jpg', 'name': 'QV.jpg' },
]

capture_image_folder = 'captured'
capture_image_path = 'captured/img_{}.jpg'
capture_interval = 10 

search_parameters = {
    'age_from': 1,
    'age_to': 100,
    'gender': ['Woman', 'Man'],
}

metadata_folder = 'metadata'
metadata_path = 'metadata/{}.json'

hr_database_folder = 'hr_database'
outcome_folder = 'outcome'

def capture_images():
    if not os.path.exists(capture_image_folder):
        os.makedirs(capture_image_folder)

    capture_stream_url = cv2.VideoCapture(0)
    start_time = time.time()
    image_count = 0

    while True:
        ret, frame = capture_stream_url.read()
        if not ret:
            print("Failed to capture image.")
            break

        cv2.imshow('Controlling my computer camera', frame)

        current_time = time.time()
        if current_time - start_time < capture_interval:
            img_path = os.path.join(capture_image_folder, f'img_{image_count}.jpg')
            cv2.imwrite(img_path, frame)
            print(f"Captured {img_path}")
            image_count += 1
            time.sleep(10)  
        else:
            break 

    capture_stream_url.release()
    cv2.destroyAllWindows()

def imagery_analysis():
    files = os.listdir(capture_image_folder)

    for file in files:
        image_file_path = os.path.join(capture_image_folder, file)
        metadata_file_name = file.split('.')[0]
        detected_faces = DeepFace.analyze(image_file_path,
                                           actions=['gender', 'age'],
                                           detector_backend='retinaface')
        if len(detected_faces) > 0:
            json_file_path = f'{metadata_folder}/{metadata_file_name}.json'
            with open(json_file_path, 'w') as json_file:
                data = {
                    'age': detected_faces[0]['age'],
                    'gender': detected_faces[0]['dominant_gender'],
                    'file': file
                }
                json.dump(data, json_file)

def deep_face_analysis_go():
    capture_images()
    imagery_analysis()
    return 'Deep Face Analyzer: DONE!'

def get_possible_rule_breakers():
    possible_rule_brakers = []

    files = os.listdir(metadata_folder)

    for file in files:
        metadata_file_path = f'{metadata_folder}/{file}'

        if os.path.isfile(metadata_file_path):
            with open(metadata_file_path, 'r') as file_metadata:
                content_file_metadata = json.load(file_metadata)
                print(f"Checking: {content_file_metadata}")  

                if (content_file_metadata['age'] >= search_parameters['age_from'] and
                    content_file_metadata['age'] <= search_parameters['age_to'] and
                    content_file_metadata['gender'] in search_parameters['gender']):
                    possible_rule_brakers.append(content_file_metadata)


    print(f"Possible Rule Breakers: {possible_rule_brakers}")  
    return possible_rule_brakers


def report_rule_breakers(possible_rule_brakers):
    print(possible_rule_brakers)

    for person in possible_rule_brakers:
        face_file_path = f'{capture_image_folder}/{person["file"]}'
        face = DeepFace.analyze(img_path=face_file_path, detector_backend='retinaface')

        if len(face) > 0:
            text = f'{person["gender"].lower()}, {person["age"]}'
            img = cv2.imread(face_file_path)

            x = face[0]['region']['x']
            y = face[0]['region']['y']
            w = face[0]['region']['w']
            h = face[0]['region']['h']

            cv2.rectangle(img, (x, y), (x + w, y + h), (204, 255, 229), 2)

            font = cv2.FONT_HERSHEY_PLAIN
            font_scale = 1.5
            font_thickness = 2
            text_color = (204, 255, 229)
            text_position = (x, y - 15)

            cv2.putText(img, text, text_position, font, font_scale, text_color, font_thickness)

            
            cv2.imshow('Face Analysis', img)
            cv2.waitKey(0)  


def get_img_rule_braker(x, y, w, h, gender, age, file_path, name, probability, verified):
    
    probability_value = float(probability.split('%')[0])
    
    
    name_display = name if probability_value >= 80 else "Unknown"

    text = f'{gender.lower()}, {age}'
    img = cv2.imread(file_path)

    cv2.rectangle(img, (x, y), (x + w, y + h), (204, 255, 229), 5)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 3
    font_thickness = 3
    text_color = (204, 255, 229)
    text_position = (x, y - 35)

    cv2.putText(img, text, text_position, font, font_scale, text_color, font_thickness)

    cv2.line(img, (x + 25, y + h + 45), (x + w - 25, y + h + 45), (204, 255, 229), 60)
    cv2.putText(img, f'{name_display}, {probability}, {verified}', 
                (x + 45, y + h + 55), font, 1, (128, 128, 128), 4)

    cv2.imshow('Rule Breaker', img)
    cv2.waitKey(0)  
    return img

def calculate_similarity(embedding1, embedding2):
    return cosine_similarity([embedding1], [embedding2])[0][0]




def search_in_hr_database(possible_rule_brakers):
    
    if not os.path.exists(outcome_folder):
        os.makedirs(outcome_folder)

    best_overall_match = None
    highest_overall_similarity = 0

    for person in possible_rule_brakers:
        person_file_path = os.path.join(capture_image_folder, person["file"])

        try:
            person_representation = DeepFace.represent(
                img_path=person_file_path,
                model_name='Facenet512',
                detector_backend='retinaface'
            )

            if not person_representation:
                print(f"No representation found for {person_file_path}")
                continue

            person_embedding = person_representation[0]["embedding"]

            highest_similarity = 0
            best_match = None

            for hr_file in hr_database_files:
                hr_file_path = hr_file['file']
                try:
                    hr_representation = DeepFace.represent(
                        img_path=hr_file_path,
                        model_name='Facenet512',
                        detector_backend='retinaface'
                    )

                    if not hr_representation:
                        print(f"No representation found for {hr_file_path}")
                        continue

                    hr_embedding = hr_representation[0]["embedding"]

                    similarity = calculate_similarity(person_embedding, hr_embedding)

                    print(f"Similarity between {person_file_path} and {hr_file_path}: {similarity:.4f}")

                    if similarity > highest_similarity:
                        highest_similarity = similarity
                        best_match = hr_file

                except Exception as e:
                    print(f"Error processing HR file {hr_file_path}: {e}")
                    continue

            if highest_similarity > highest_overall_similarity:
                highest_overall_similarity = highest_similarity
                best_overall_match = {
                    'person': person,
                    'best_match': best_match,
                    'similarity': highest_similarity
                }

        except Exception as e:
            print(f"Error processing person {person_file_path}: {e}")

    if best_overall_match:
        person = best_overall_match['person']
        best_match = best_overall_match['best_match']
        highest_similarity = best_overall_match['similarity']

        person_file_path = os.path.join(capture_image_folder, person["file"])
        print(f"\nHighest similarity: {highest_similarity:.4f}")
        
        if highest_similarity >= 0.49:
            print(f"Matched HR name: {best_match['name']}")
            person_name = best_match['name'].split('.')[0].replace('_', ' ')
        else:
            print(f"Unknown")
            person_name = "Unknown"

        print(f"Similarity percentage: {highest_similarity * 100:.2f}%")

        person_verified = 'VERIFIED'
        person_probability = f"{highest_similarity * 100:.2f}% similarity"
        
        img = cv2.imread(person_file_path)
        if img is None:
            print(f"Failed to load image {person_file_path}")
            return

        text = f'{person["gender"].lower()}, {person["age"]} - {person_name}, {person_verified}, {person_probability}'
        cv2.putText(img, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        try:
            face = DeepFace.analyze(img_path=person_file_path, detector_backend='retinaface')
            if face and len(face) > 0:
                x = face[0]['region']['x']
                y = face[0]['region']['y']
                w = face[0]['region']['w']
                h = face[0]['region']['h']
                cv2.rectangle(img, (x, y), (x + w, y + h), (204, 255, 229), 2)
        except Exception as e:
            print(f"Error drawing rectangle on {person_file_path}: {e}")

        outcome_image_path = os.path.join(outcome_folder, f"verified_{person['file']}")
        cv2.imwrite(outcome_image_path, img)
        print(f"Verified image saved to {outcome_image_path}")

        cv2.imshow("Verified Person", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No matches found with a high enough similarity.")

def deep_face_analysis_go():
    capture_images()
    imagery_analysis()

    poi_list = get_possible_rule_breakers()
    report_rule_breakers(poi_list)
    search_in_hr_database(poi_list)

    return 'Deep Face Analyzer: DONE!'

if __name__ == '__main__':
    deep_face_analysis_go()
