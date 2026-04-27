import cv2
import numpy as np
from mtcnn import MTCNN
import os
import shutil
from skimage import feature
from scipy.spatial.distance import cosine

# Initialize face detector
detector = MTCNN()

# Database path
DATASET_PATH = "model/known_faces/"
os.makedirs(DATASET_PATH, exist_ok=True)

class FaceDetector:
    def __init__(self):
        self.detector = MTCNN()

    def detect_and_extract(self, image, min_face_size=50):
        """Detect faces with improved accuracy"""
        if isinstance(image, str):
            image = cv2.imread(image)
            if image is None:
                return [], [], None
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        original_img = image.copy()

        try:
            faces_data = self.detector.detect_faces(image)
        except:
            return [], [], original_img

        extracted_faces = []
        bboxes = []

        valid_faces = []
        for face in faces_data:
            x, y, w, h = face['box']
            confidence = face['confidence']

            if confidence > 0.95 and w > min_face_size and h > min_face_size:
                valid_faces.append(face)

        valid_faces.sort(key=lambda f: f['box'][2] * f['box'][3], reverse=True)

        for face in valid_faces:
            x, y, w, h = face['box']
            confidence = face['confidence']

            padding = int(min(w, h) * 0.2)
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(image.shape[1], x + w + padding)
            y2 = min(image.shape[0], y + h + padding)

            face_img = image[y1:y2, x1:x2]

            if face_img.size > 0 and face_img.shape[0] > 50 and face_img.shape[1] > 50:
                face_resized = cv2.resize(face_img, (160, 160))
                extracted_faces.append(face_resized)
                bboxes.append((x1, y1, x2, y2, confidence))

        return extracted_faces, bboxes, original_img

class FaceRecognizer:
    def __init__(self):
        self.known_faces_dir = DATASET_PATH
        self.known_face_features = []
        self.known_names = []
        self.MATCH_THRESHOLD = 0.62
        self.load_known_faces()

    def extract_face_features(self, face_img):
        """Extract comprehensive face features"""
        try:
            gray = cv2.cvtColor(face_img, cv2.COLOR_RGB2GRAY)
            gray = cv2.resize(gray, (160, 160))
            
            hog_features = feature.hog(gray, orientations=9, pixels_per_cell=(8, 8),
                                       cells_per_block=(2, 2), visualize=False)
            
            from skimage.feature import local_binary_pattern
            radius = 3
            n_points = 8 * radius
            lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
            lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), 
                                       density=True)
            
            hsv = cv2.cvtColor(face_img, cv2.COLOR_RGB2HSV)
            color_hist = []
            for i in range(3):
                hist = cv2.calcHist([hsv], [i], None, [32], [0, 256])
                hist = cv2.normalize(hist, hist).flatten()
                color_hist.extend(hist)
            
            edges = cv2.Canny(gray, 50, 150)
            edge_hist = cv2.calcHist([edges], [0], None, [32], [0, 256])
            edge_hist = cv2.normalize(edge_hist, edge_hist).flatten()
            
            features = np.concatenate([hog_features, lbp_hist, color_hist, edge_hist])
            features = features / (np.linalg.norm(features) + 1e-6)
            
            return features
        except Exception as e:
            print(f"Feature extraction error: {e}")
            return None

    def load_known_faces(self):
        """Load all known faces from database"""
        self.known_face_features = []
        self.known_names = []
        
        if not os.path.exists(self.known_faces_dir):
            return

        for person_name in os.listdir(self.known_faces_dir):
            person_dir = os.path.join(self.known_faces_dir, person_name)
            if os.path.isdir(person_dir):
                for img_file in os.listdir(person_dir):
                    if img_file.endswith(('.jpg', '.png', '.jpeg')):
                        img_path = os.path.join(person_dir, img_file)
                        try:
                            img = cv2.imread(img_path)
                            if img is not None:
                                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                img = cv2.resize(img, (160, 160))
                                features = self.extract_face_features(img)
                                if features is not None:
                                    self.known_face_features.append(features)
                                    self.known_names.append(person_name)
                        except Exception as e:
                            print(f"Error loading {img_path}: {e}")
                            continue

    def recognize(self, face_img):
        """Recognize face with strict thresholds"""
        if not self.known_names:
            return "Unknown", 0.0

        if face_img is None:
            return "Unknown", 0.0

        input_features = self.extract_face_features(face_img)
        
        if input_features is None:
            return "Unknown", 0.0
        
        scores = []
        for known_features in self.known_face_features:
            similarity = 1 - cosine(input_features, known_features)
            scores.append(similarity)
        
        best_score = max(scores) if scores else 0.0
        best_match_idx = np.argmax(scores) if scores else -1
        best_match = self.known_names[best_match_idx] if best_match_idx >= 0 else "Unknown"
        
        sorted_scores = sorted(scores, reverse=True)
        second_best = sorted_scores[1] if len(sorted_scores) > 1 else 0.0
        score_difference = best_score - second_best
        
        if best_score >= self.MATCH_THRESHOLD and score_difference >= 0.08:
            return best_match, best_score
        else:
            return "Unknown", best_score

    def add_face(self, person_name, face_img):
        """Add new face to database - returns boolean and count"""
        try:
            person_name = person_name.strip().lower()
            person_dir = os.path.join(self.known_faces_dir, person_name)
            os.makedirs(person_dir, exist_ok=True)

            # Get current count
            img_count = len([f for f in os.listdir(person_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
            
            # Save image
            import time
            timestamp = int(time.time())
            img_path = os.path.join(person_dir, f"face_{img_count+1}_{timestamp}.jpg")
            
            face_bgr = cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR)
            success = cv2.imwrite(img_path, face_bgr)
            
            if success:
                self.load_known_faces()
                return True, img_count + 1  # Return tuple (success, count)
            else:
                return False, 0
        except Exception as e:
            print(f"Error saving face: {e}")
            return False, 0
    
    def delete_person(self, person_name):
        person_name = person_name.strip().lower()
        person_dir = os.path.join(self.known_faces_dir, person_name)
        
        if os.path.exists(person_dir):
            try:
                shutil.rmtree(person_dir, ignore_errors=True)
                self.load_known_faces()
                return True
            except:
                return False
        return False
    
    def get_all_people(self):
        people = []
        if os.path.exists(self.known_faces_dir):
            for person in os.listdir(self.known_faces_dir):
                person_path = os.path.join(self.known_faces_dir, person)
                if os.path.isdir(person_path):
                    img_count = len([f for f in os.listdir(person_path) if f.endswith(('.jpg', '.png', '.jpeg'))])
                    if img_count > 0:
                        people.append(person)
        return people

class DeepfakeDetector:
    def __init__(self):
        pass

    def predict(self, face_img):
        """Detect if face is real or fake"""
        if face_img is None:
            return False, 0.0

        gray = cv2.cvtColor(face_img, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape

        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        left_half = gray[:, :w//2]
        right_half = np.fliplr(gray[:, w//2:])
        if left_half.shape == right_half.shape:
            symmetry_score = np.mean(np.abs(left_half - right_half)) / 255.0
        else:
            symmetry_score = 0.5

        skin_std = np.std(gray) / 255.0
        
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (h * w)

        real_score = (
            (laplacian_var / 200) * 0.4 +
            (1 - symmetry_score) * 0.3 +
            (1 - skin_std) * 0.2 +
            edge_density * 0.1
        )

        real_score = min(0.95, max(0.05, real_score))
        is_real = real_score > 0.5
        confidence = real_score if is_real else 1 - real_score

        return is_real, confidence

# Initialize modules
face_detector = FaceDetector()
face_recognizer = FaceRecognizer()
deepfake_detector = DeepfakeDetector()

# Public functions
def detect_face(image):
    faces, boxes, _ = face_detector.detect_and_extract(image)
    simple_boxes = [(x1, y1, x2, y2) for (x1, y1, x2, y2, conf) in boxes]
    return faces, simple_boxes

def recognize_face(face_img):
    name, confidence = face_recognizer.recognize(face_img)
    return name, confidence

def predict_deepfake(face_img):
    is_real, confidence = deepfake_detector.predict(face_img)
    if is_real:
        return "REAL", confidence
    else:
        return "FAKE", confidence

def add_new_face(name, face_img):
    """Returns (success, count) tuple"""
    return face_recognizer.add_face(name, face_img)

def delete_face(person_name):
    return face_recognizer.delete_person(person_name)

def get_all_faces():
    return face_recognizer.get_all_people()