# Real-time Face Recognition with Eigenfaces using OpenCV and SVM

# Membutuhkan packages yang ada di requirements.txt
# Untuk menginstall dependencies, jalankan perintah berikut di terminal:
#pip install -r requirements.txt

# Import library yang dibutuhkan
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
import pickle

# Fungsi untuk load dan konversi gambar ke grayscale
def load_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f'Error: Could not load image {image_path}')
        return None, None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image, gray

# Load dataset dari folder lokal
dataset_dir = 'images'
images = []
labels = []
for root, dirs, files in os.walk(dataset_dir):
    for f in files:
        _, image = load_image(os.path.join(root, f))
        if image is None:
            continue
        images.append(image)
        labels.append(os.path.basename(root)) #nama foder sebagai label

print(f'Total data: {len(labels)}')

# Face detection menggunakan Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Fungsi untuk mendeteksi wajah pada gambar
def detect_faces(image_gray, scale_factor=1.1, min_neighbors=5, min_size=(30, 30)):
    faces = face_cascade.detectMultiScale(
        image_gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=min_size
    )
    return faces

# Fungsi untuk crop wajah dari gambar berdasarkan koordinat deteksi
def crop_faces(image_gray, faces, return_all=False):
    cropped_faces = []
    selected_faces = []
    if len(faces) > 0:
        if return_all:
            for x, y, w, h in faces:
                selected_faces.append((x, y, w, h))
                cropped_faces.append(image_gray[y:y+h, x:x+w])
        else:
            x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
            selected_faces.append((x, y, w, h))
            cropped_faces.append(image_gray[y:y+h, x:x+w])
    return cropped_faces, selected_faces

# Resize dan flatten
face_size = (128, 128)

def resize_and_flatten(face):
    face_resized = cv2.resize(face, face_size)
    face_flattened = face_resized.flatten()
    return face_flattened

# Preprocessing data
# Ekstraksi fitur dari gambar wajah
X = []
y = []

for image, label in zip(images, labels):
    faces = detect_faces(image)
    cropped_faces, _ = crop_faces(image, faces)
    if len(cropped_faces) > 0:
        face_flattened = resize_and_flatten(cropped_faces[0])
        X.append(face_flattened)
        y.append(label)

X = np.array(X)
y = np.array(y)
print(f'Feature shape: {X.shape}')

# Split data untuk training dan testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=177, stratify=y)

# Transformet untuk Mean Centering
class MeanCentering(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.mean_face = np.mean(X, axis=0)
        return self
    def transform(self, X):
        return X - self.mean_face


# Pipeline: mean centering → PCA → SVM
pipe = Pipeline([
    ('centering', MeanCentering()),
    ('pca', PCA(svd_solver='randomized', whiten=True, random_state=177)),
    ('svc', SVC(kernel='linear', random_state=177))
])

# Training model
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

# Evaluasi performa model
print(classification_report(y_test, y_pred))

# Simpan pipeline model ke file .pkl
with open('eigenface_pipeline.pkl', 'wb') as f:
    pickle.dump(pipe, f)

# Fungsi untuk mengambil skor dari SVM
def get_eigenface_score(X):
    X_pca = pipe[:2].transform(X)
    eigenface_scores = np.max(pipe[2].decision_function(X_pca), axis=1)
    return eigenface_scores

# Fungsi untuk memprediksi wajah baru
def eigenface_prediction(image_gray):
    faces = detect_faces(image_gray)
    cropped_faces, selected_faces = crop_faces(image_gray, faces)
    if len(cropped_faces) == 0:
        return [], [], []
    X_face = []
    for face in cropped_faces:
        face_flattened = resize_and_flatten(face)
        X_face.append(face_flattened)
    X_face = np.array(X_face)
    labels = pipe.predict(X_face)
    scores = get_eigenface_score(X_face)
    return scores, labels, selected_faces

# Visualisasi hasil
# Fungsi untuk menggambar teks dan skor pada gambar
def draw_text(image, label, score,
              font=cv2.FONT_HERSHEY_SIMPLEX,
              pos=(0, 0),
              font_scale=0.6,
              font_thickness=2,
              text_color=(0, 0, 0),
              text_color_bg=(0, 255, 0)):
    x, y = pos
    score_text = f'Score: {score:.2f}'
    (w1, h1), _ = cv2.getTextSize(score_text, font, font_scale, font_thickness)
    (w2, h2), _ = cv2.getTextSize(label, font, font_scale, font_thickness)
    cv2.rectangle(image, (x, y-h1-h2-25), (x + max(w1, w2)+20, y), text_color_bg, -1)
    cv2.putText(image, label, (x+10, y-10), font, font_scale, text_color, font_thickness)
    cv2.putText(image, score_text, (x+10, y-h2-15), font, font_scale, text_color, font_thickness)

# Fungsi untuk menggambar hasil deteksi wajah
def draw_result(image, scores, labels, coords):
    result_image = image.copy()
    for (x, y, w, h), label, score in zip(coords, labels, scores):
        cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        draw_text(result_image, label, score, pos=(x, y))
    return result_image

# Real-time face recognition dari webcam
cap = cv2.VideoCapture(0)  # 0 = default webcam

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    scores, labels, coords = eigenface_prediction(gray_frame)

    # Jika ada wajah yang terdeteksi, gambar hasilnya
    if scores and labels and coords:
        frame = draw_result(frame, scores, labels, coords)

    # Tampilkan hasil deteksi wajah
    cv2.imshow('Real-Time Face Recognition', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27 or key == ord('q'):  # Tekan ESC atau 'q' untuk keluar
        break

# Bersihkan resources
cap.release()
cv2.destroyAllWindows()

# Visualisasi hasil prediksi pada data test
plt.figure(figsize=(16, 10))
rows, cols = 2, 5  # Grid 2x5
max_images = rows * cols  # Maksimal 10 gambar

for i in range(min(len(X_test), max_images)):  # Batasi iterasi hingga max_images
    img_gray = X_test[i].reshape(face_size)
    img = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    
    sample_scores, sample_labels, sample_faces = eigenface_prediction(img_gray)
    
    # Pengecekan untuk memastikan sample_labels tidak kosong
    if sample_labels:
        result_image = draw_result(img, sample_scores, sample_labels, sample_faces)
        plt.subplot(rows, cols, i + 1)  # Sesuaikan grid menjadi 2x5
        plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
        plt.title(f"True: {y_test[i]}\nPred: {sample_labels[0]}", fontsize=8)
    else:
        plt.subplot(rows, cols, i + 1)  # Sesuaikan grid menjadi 2x5
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(f"True: {y_test[i]}\nPred: None", fontsize=8)
    
    plt.axis("off")

plt.tight_layout()
plt.show()