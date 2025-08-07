import os
import pickle
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from datetime import datetime, date
from django.conf import settings  # type: ignore
from django.http import JsonResponse, HttpResponse  # type: ignore
from django.shortcuts import render  # type: ignore
from django.views.decorators.csrf import csrf_exempt  # type: ignore
from django.core.files.storage import FileSystemStorage  # type: ignore
from keras_facenet import FaceNet  # type: ignore
from scipy.spatial.distance import cosine  # type: ignore
import cv2  # type: ignore
import json

# Initialize FaceNet
embedder = FaceNet()

# Load known face encodings
pickle_path = os.path.join(settings.BASE_DIR, 'face', 'face_encodings.pkl')
with open(pickle_path, 'rb') as f:
    data = pickle.load(f)
known_encodings = data['encodings']
known_names = data['names']

# Prepare attendance file
attendance_file = os.path.join(settings.BASE_DIR, 'face', 'attendance.csv')
if not os.path.exists(attendance_file):
    pd.DataFrame(columns=["StudentID", "Date", "Time"]).to_csv(attendance_file, index=False)

# Mark attendance
def mark_attendance(student_id):
    df = pd.read_csv(attendance_file)
    today_str = date.today().strftime("%Y-%m-%d")
    if not ((df["StudentID"] == student_id) & (df["Date"] == today_str)).any():
        now = datetime.now()
        new_row = pd.DataFrame([[student_id, today_str, now.strftime("%H:%M:%S")]],
                               columns=["StudentID", "Date", "Time"])
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(attendance_file, index=False)

# Home
def home(request):
    return render(request, 'homep.html')

# Start attendance
@csrf_exempt
def start_attendance(request):
    if request.method == 'POST':
        if 'image' in request.FILES:
            image_file = request.FILES['image']
            fs = FileSystemStorage()
            filename = fs.save(image_file.name, image_file)
            image_path = fs.path(filename)

            img = cv2.imread(image_path)
            if img is None:
                return JsonResponse({'error': 'Invalid image'}, status=400)

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            detections = embedder.extract(img_rgb, threshold=0.95)

            if not detections:
                return JsonResponse({'name': 'Unknown', 'confidence': 0, 'marked': False})

            best_id = "Unknown"
            best_score = 1.0
            for det in detections:
                embedding = det["embedding"]
                for known_embedding, known_id in zip(known_encodings, known_names):
                    dist = cosine(embedding, known_embedding)
                    if dist < best_score:
                        best_score = dist
                        best_id = known_id

            confidence = int((1 - best_score) * 100)

            if best_score < 0.25:
                mark_attendance(best_id)
                return JsonResponse({'name': best_id, 'confidence': confidence, 'marked': True})
            else:
                return JsonResponse({'name': 'Not Found', 'confidence': confidence, 'marked': False})

        return JsonResponse({'error': 'No image uploaded'}, status=400)

    return render(request, 'attendence.html')

# Dashboard
def dashboard(request):
    dataset_path = os.path.join(settings.BASE_DIR, 'face', 'dataset')
    all_ids = sorted([
        folder for folder in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, folder))
    ])

    selected_date_str = request.GET.get('date')
    selected_date = date.today()
    if selected_date_str:
        try:
            selected_date = datetime.strptime(selected_date_str, "%Y-%m-%d").date()
        except ValueError:
            pass

    if os.path.exists(attendance_file):
        df = pd.read_csv(attendance_file)
        try:
            df['Date'] = pd.to_datetime(df['Date'], format="%Y-%m-%d").dt.date
        except Exception:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce').dt.date

        df_today = df[df['Date'] == selected_date]
        present_ids = sorted(df_today['StudentID'].astype(str).unique())
    else:
        present_ids = []

    if 'df_today' not in locals() or df_today.empty:
        present_ids = []
        absent_ids = []
        no_data = True
    else:
        absent_ids = sorted(list(set(all_ids) - set(present_ids)))
        no_data = False

    return render(request, 'dashboard.html', {
        'today': selected_date,
        'present_ids': present_ids,
        'absent_ids': absent_ids,
        'selected_date_str': selected_date.strftime("%Y-%m-%d"),
        'no_data': no_data
    })

# Download attendance summary
def download_attendance_summary(request):
    dataset_path = os.path.join(settings.BASE_DIR, 'face', 'dataset')
    attendance_path = attendance_file

    all_rolls = set(os.listdir(dataset_path))
    attendance = pd.read_csv(attendance_path)
    present_rolls = set(attendance['StudentID'])

    absent_rolls = all_rolls - present_rolls

    present_list = sorted(present_rolls)
    absent_list = sorted(absent_rolls)
    max_len = max(len(present_list), len(absent_list))
    present_list.extend([''] * (max_len - len(present_list)))
    absent_list.extend([''] * (max_len - len(absent_list)))

    df = pd.DataFrame({
        'Present': present_list,
        'Absent': absent_list
    })

    import io
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Attendance Summary')
    output.seek(0)

    response = HttpResponse(
        output,
        content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )
    response['Content-Disposition'] = 'attachment; filename=attendance_summary.xlsx'
    return response

# 404
def not_found(request):
    return render(request, 'notfound.html')
