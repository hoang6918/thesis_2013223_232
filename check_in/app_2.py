from flask import Flask, render_template, Response, send_from_directory
import cv2
from deepface import DeepFace
import numpy as np
from mtcnn import MTCNN
from sklearn.metrics.pairwise import cosine_similarity
from deepface.commons import functions
import sqlite3
from datetime import datetime
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from flask_cors import CORS
import pyttsx3
import time
import threading




app = Flask(__name__)
CORS(app)

credentials_file = "river-engine-400013-b1d721c87331.json"
spreadsheet_id = "1qTj4g-Yy-iuscuGuyt38GCMTPKctLtZhC1s5e0qfmU8"
worksheet_name = "database"
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
credentials = ServiceAccountCredentials.from_json_keyfile_name(credentials_file, scope)
client = gspread.authorize(credentials)
sheet = client.open_by_key(spreadsheet_id).worksheet(worksheet_name)


# Connect to the SQLite database with check_same_thread set to False
conn = sqlite3.connect("face_encodings_deepface_2.db", check_same_thread=False)
cursor = conn.cursor()
face_names = []
sent_names = set()

now = datetime.now()
current_date = now.strftime("%Y-%m-%d")
# Global variable to store the latest recognized information
latest_recognized_info_default = None
latest_recognized_info_rtsp = None


def update_checkbox(sheet, row, column, value):
    cell = sheet.cell(row, column)
    cell.value = value
    sheet.update_cells([cell])


def calculate_similarity(embedding1, embedding2):
    # Convert the embeddings to NumPy arrays
    embedding1 = np.array(embedding1)
    embedding2 = np.array(embedding2)
    
    # Calculate the cosine similarity between the embeddings
    similarity = cosine_similarity(embedding1.reshape(1, -1), embedding2.reshape(1, -1))
    return similarity[0][0]

def calculate_confidence(similarity):
    # Calculate the confidence score based on the cosine similarity
    confidence = similarity
    return confidence
    
def speak_greeting(name):
        def run_speak():
            engine = pyttsx3.init()
            volume = engine.getProperty('volume')
            engine.setProperty('volume', volume+1)
            rate = engine.getProperty('rate')
            engine.setProperty('rate', rate-100)
            voices = engine.getProperty('voices')
            vi_voice_id = None
            for voice in voices:
                if "Microsoft An - Vietnamese (Vietnam)" in voice.name:
                    vi_voice_id = voice.id
                    break
            if vi_voice_id:
                engine.setProperty('voice', vi_voice_id)
            else:
                print("Vietnamese voice not found. Using default voice.")
            greeting = f"Chào mừng bạn {name} đã đến tham dự hội nghị ngày hôm nay"
            engine.say(greeting)
            engine.runAndWait()
        
        # Run the text-to-speech operation in a separate thread
        threading.Thread(target=run_speak).start()



def recognize_face_default(face_embedding):
    global latest_recognized_info_default
    cursor.execute("SELECT encoding, name, id_code, department, class_name FROM face_encodings_deepface")
    rows = cursor.fetchall()
    max_similarity = -1
    recognized_name = "Unknown"
    recognized_id_code = "Unknown"
    recognized_department = "Unknown"
    recognized_class_name = "Unknown"
    # Iterate over the retrieved rows
    for row in rows:
        # Retrieve the stored embedding and name
        stored_embedding = np.frombuffer(row[0], dtype=np.float64)
        stored_name = row[1]
        stored_id_code = row[2]
        stored_department = row[3]
        stored_class_name = row[4]
        # Calculate the cosine similarity between the face embedding from the camera and the stored embedding
        similarity = calculate_similarity(face_embedding, stored_embedding)
        # Update the recognized name if the similarity is greater than the maximum similarity so far
        if similarity > max_similarity:
            max_similarity = similarity
            recognized_name = stored_name
            recognized_id_code = stored_id_code
            recognized_department = stored_department
            recognized_class_name = stored_class_name
    # Calculate the confidence score based on the similarity
    confidence = calculate_confidence(max_similarity)
    # Adjust the threshold for considering a face as "Unknown"
    if max_similarity < 0.8:
        recognized_id_code = "Unknown" 
        latest_recognized_info_default = {"name": "Unknown", "id_code": recognized_id_code, "department": "Unknown", "class_name": "Unknown"}
        return recognized_name, confidence, recognized_id_code 
    else:
        latest_recognized_info_default = {"name": recognized_name, "id_code": recognized_id_code, "department": recognized_department, "class_name": recognized_class_name}  
        # Print the latest recognized information to the terminal
        print("Latest Recognized Info:", latest_recognized_info_default)
        return recognized_name, confidence, recognized_id_code
    

def recognize_face_rtsp(face_embedding):
    global latest_recognized_info_rtsp
    cursor.execute("SELECT encoding, name, id_code, department, class_name FROM face_encodings_deepface")
    rows = cursor.fetchall()
    max_similarity = -1
    recognized_name = "Unknown"
    recognized_id_code = "Unknown"
    recognized_department = "Unknown"
    recognized_class_name = "Unknown"
    # Iterate over the retrieved rows
    for row in rows:
        # Retrieve the stored embedding and name
        stored_embedding = np.frombuffer(row[0], dtype=np.float64)
        stored_name = row[1]
        stored_id_code = row[2]
        stored_department = row[3]
        stored_class_name = row[4]
        # Calculate the cosine similarity between the face embedding from the camera and the stored embedding
        similarity = calculate_similarity(face_embedding, stored_embedding)
        # Update the recognized name if the similarity is greater than the maximum similarity so far
        if similarity > max_similarity:
            max_similarity = similarity
            recognized_name = stored_name
            recognized_id_code = stored_id_code
            recognized_department = stored_department
            recognized_class_name = stored_class_name
    # Calculate the confidence score based on the similarity
    confidence = calculate_confidence(max_similarity)
    # Adjust the threshold for considering a face as "Unknown"
    if max_similarity < 0.8:
        recognized_id_code = "Unknown" 
        latest_recognized_info_rtsp = {"name": "Unknown", "id_code": recognized_id_code, "department": "Unknown", "class_name": "Unknown"}
        return recognized_name, confidence, recognized_id_code 
    else:
        latest_recognized_info_rtsp = {"name": recognized_name, "id_code": recognized_id_code, "department": recognized_department, "class_name": recognized_class_name}  
        # Print the latest recognized information to the terminal
        print("Latest Recognized Info:", latest_recognized_info_rtsp)
        return recognized_name, confidence, recognized_id_code

# Hàng và cột bắt đầu trong Google Sheets
start_row_sheets = 6
start_column_sheets = 1

# Cập nhật một ô checkbox trong Google Sheets
def update_checkbox(sheet, row, column, value):
    cell = sheet.cell(row, column)
    cell.value = value
    sheet.update_cells([cell])
    
last_recognized_name = None             

# Hàm để lấy vị trí ô trống tiếp theo trong cột
def get_next_empty_row(sheet, column):
    values = sheet.col_values(column)
    next_empty_row = start_row_sheets + len(values)
    return next_empty_row

# Hàm để push danh sách recognized_name trước khi nhận diện
def push_names_to_sheet(sheet, names):
    # Xóa dữ liệu cũ trong cột NGÀY TỔ CHỨC, THỜI ĐIỂM CHECK IN và đặt checkbox về False
    cell_range = sheet.range(start_row_sheets, start_column_sheets + 1, start_row_sheets + len(names) - 1, start_column_sheets + 2)
    for cell in cell_range:
        cell.value = ""
    checkbox_range = sheet.range(start_row_sheets, start_column_sheets + 3, start_row_sheets + len(names) - 1, start_column_sheets + 3)
    for checkbox_cell in checkbox_range:
        checkbox_cell.value = False

    # Ghi danh sách names vào cột STT và MSSV
    for i, name in enumerate(names):
        sheet.update_cell(start_row_sheets + i, start_column_sheets, i + 1)  # Cập nhật STT
        sheet.update_cell(start_row_sheets + i, start_column_sheets + 1, name)  # Cập nhật MSSV

# Lấy danh sách recognized_name từ SQLite và push vào Google Sheets
# After fetching id_code from the database
cursor.execute("SELECT id_code FROM face_encodings_deepface")
rows = cursor.fetchall()
recognized_id_code = [row[0] for row in rows]

# Push id_code to Google Sheet
push_names_to_sheet(sheet, recognized_id_code)

@app.route('/captured_face/<id_code>')
def captured_face(id_code):
    return send_from_directory('static', f'{id_code}.jpg')
@app.route('/latest-info')
def latest_info():
    global latest_recognized_info_rtsp
    global latest_recognized_info_default
    return {"info_default": latest_recognized_info_default, "info_rtsp": latest_recognized_info_rtsp}
@app.route('/')
def index():
    global latest_recognized_info_rtsp
    global latest_recognized_info_default
    
    return render_template('index2.html', info_default = latest_recognized_info_default, info_rtsp = latest_recognized_info_rtsp)


#cam_path1 = 'rtsp://admin:Hcmut@k20@192.168.137.200:554/'
#cam_path2 = 'rtsp://admin:Hcmut@k20@192.168.137.200:554/'

def gen_frames_default():
    
    # Attempt to use DirectShow as backend for video capture
    camera = cv2.VideoCapture('http://192.168.0.107:8080/video')
    model = DeepFace.build_model("VGG-Face")
    face_detector = MTCNN()
    recognized_id_code_default = "Unknown"
    recognized_name_default = "Unknown"
    frame_skip = 25  # Process every 25th frame
    frame_count = 0
    while True:
        global latest_recognized_info_default
        success, frame = camera.read()
        if not success:
            break
        
        frame_count += 1
        if frame_count % frame_skip != 0:
            continue
        else:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = face_detector.detect_faces(frame_rgb)
            if len(faces) == 0:
                    latest_recognized_info_default = {"name": "warning", "id_code": "warning", "department": "warning", "class_name": "warning"}
            for face in faces:
                x, y, w, h = face['box']
                face_region = frame_rgb[y:y+h, x:x+w]
                preprocessed_face = functions.preprocess_face(face_region, target_size=(224, 224), enforce_detection=False)
                embedding = model.predict(np.expand_dims(preprocessed_face, axis=0))[0]
                if len(embedding) > 0:
                    face_embedding = embedding
                    recognized_name_default, confidence_default, recognized_id_code_default = recognize_face_default(face_embedding)
                    # Check if the recognized name is "Unknown" and adjust the text and color accordingly
                    if recognized_id_code_default == "Unknown":
                        text_color = (0, 0, 255)  # Red color for "Unknown"
                        text = "Unknown"
                        cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), text_color, 2)
                    else:
                        cv2.imwrite(f"static/{recognized_id_code_default}.jpg", frame)
                        text_color = (0, 255, 0)  # Green color for recognized faces
                        text = f"{recognized_id_code_default}: {confidence_default:.2f}"
                        cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), text_color, 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            if recognized_id_code_default not in sent_names and recognized_id_code_default != "Unknown":
                    # Handle known person check-in logic
                    time.sleep(1)
                    speak_greeting(recognized_name_default)
                    current_time = now.strftime("%H:%M:%S")
                    cell = sheet.find(recognized_id_code_default, in_column=start_column_sheets + 1)
                    if cell:
                        row_of_recognized_id_code = cell.row
                        sheet.update_cell(row_of_recognized_id_code, start_column_sheets + 2, current_date)
                        sheet.update_cell(row_of_recognized_id_code, start_column_sheets + 3, current_time)
                        checkbox_column = start_column_sheets + 4
                        update_checkbox(sheet, row_of_recognized_id_code, checkbox_column, True)
                        sent_names.add(recognized_id_code_default)
                    else:
                        next_empty_row = get_next_empty_row(sheet, start_column_sheets)
                        sheet.update_cell(next_empty_row, start_column_sheets, next_empty_row - start_row_sheets + 1)
                        sheet.update_cell(next_empty_row, start_column_sheets + 1, recognized_id_code)
                        sheet.update_cell(next_empty_row, start_column_sheets + 2, current_date)
                        sheet.update_cell(next_empty_row, start_column_sheets + 3, current_time)
                        checkbox_column = start_column_sheets + 4
                        update_checkbox(sheet, next_empty_row, checkbox_column, True)
                        sent_names.add(recognized_id_code_default)


def gen_frames_rtsp():
    
    camera = cv2.VideoCapture('http://192.168.0.106:8080/video')
    model = DeepFace.build_model("VGG-Face")
    face_detector = MTCNN()
    recognized_id_code_rtsp = "Unknown"
    recognized_name_rtsp = "Unknown"
    frame_skip = 25  # Process every 25th frame
    frame_count = 0
    while True:
        global latest_recognized_info_rtsp
        success, frame = camera.read()
        if not success:
            break
        
        frame_count += 1
        if frame_count % frame_skip != 0:
            continue
        
        else:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = face_detector.detect_faces(frame_rgb)
            if len(faces) == 0:
                    latest_recognized_info_rtsp = {"name": "warning", "id_code": "warning", "department": "warning", "class_name": "warning"}
            for face in faces:
                x, y, w, h = face['box']
                face_region = frame_rgb[y:y+h, x:x+w]
                preprocessed_face = functions.preprocess_face(face_region, target_size=(224, 224), enforce_detection=False)
                embedding = model.predict(np.expand_dims(preprocessed_face, axis=0))[0]
                if len(embedding) > 0:
                    face_embedding = embedding
                    recognized_name_rtsp, confidence_rtsp, recognized_id_code_rtsp = recognize_face_rtsp(face_embedding)
                    # Check if the recognized name is "Unknown" and adjust the text and color accordingly
                    if recognized_id_code_rtsp == "Unknown":
                        text_color = (0, 0, 255)  # Red color for "Unknown"
                        text = "Unknown"
                        cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), text_color, 2)
                    else:
                        cv2.imwrite(f"static/{recognized_id_code_rtsp}.jpg", frame)
                        text_color = (0, 255, 0)  # Green color for recognized faces
                        text = f"{recognized_id_code_rtsp}: {confidence_rtsp:.2f}"
                        cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), text_color, 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            if recognized_id_code_rtsp not in sent_names and recognized_id_code_rtsp != "Unknown":
                    # Handle known person check-in logic
                    time.sleep(1)
                    speak_greeting(recognized_name_rtsp)
                    current_time = now.strftime("%H:%M:%S")
                    cell = sheet.find(recognized_id_code_rtsp, in_column=start_column_sheets + 1)
                    if cell:
                        row_of_recognized_id_code = cell.row
                        sheet.update_cell(row_of_recognized_id_code, start_column_sheets + 2, current_date)
                        sheet.update_cell(row_of_recognized_id_code, start_column_sheets + 3, current_time)
                        checkbox_column = start_column_sheets + 4
                        update_checkbox(sheet, row_of_recognized_id_code, checkbox_column, True)
                        sent_names.add(recognized_id_code_rtsp)
                    else:
                        next_empty_row = get_next_empty_row(sheet, start_column_sheets)
                        sheet.update_cell(next_empty_row, start_column_sheets, next_empty_row - start_row_sheets + 1)
                        sheet.update_cell(next_empty_row, start_column_sheets + 1, recognized_id_code)
                        sheet.update_cell(next_empty_row, start_column_sheets + 2, current_date)
                        sheet.update_cell(next_empty_row, start_column_sheets + 3, current_time)
                        checkbox_column = start_column_sheets + 4
                        update_checkbox(sheet, next_empty_row, checkbox_column, True)
                        sent_names.add(recognized_id_code_rtsp)

@app.route('/video_feed_default')
def video_feed_default():  
    return Response(gen_frames_default(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_rtsp')
def video_feed_rtsp():  
    return Response(gen_frames_rtsp(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)