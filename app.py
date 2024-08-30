import csv
import cv2
import os
import mysql.connector
from flask import Flask, render_template, redirect, url_for, session, request, flash
from datetime import date, datetime
import numpy as np
import pandas as pd
from mysql.connector import Error
from PIL import Image

# Defining Flask App
app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Number of images to take for each user
nimgs = 50

# Lưu ngày hôm nay theo 2 định dạng khác nhau
datetoday = date.today().strftime("%m_%d_%y")

# Khởi tạo đối tượng VideoCapture để truy cập WebCam
cascade_path = 'haarcascade_frontalface_default.xml'
face_detector = cv2.CascadeClassifier(cascade_path)

# Kiểm tra và tạo thư mục nếu chưa tồn tại
os.makedirs('Attendance', exist_ok=True)

os.makedirs('dataset', exist_ok=True)
os.makedirs('trainer', exist_ok=True)

# Initialize attendance CSV file
if f'Attendance-{datetoday}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-{datetoday}.csv', 'w') as f:
        f.write('Name,Roll,Time')

# Khởi tạo nhận diện khuôn mặt và camera
recognizer = cv2.face.LBPHFaceRecognizer_create()
cam = cv2.VideoCapture(0)


# Get total registered users
def readnames():
    with open('names.txt', 'r') as f:
        return f.read().splitlines()

def writenames(arr):
    with open('names.txt', 'w') as f:
        f.write('\n'.join(arr))

namearr = readnames()

def check_mysql_connection():
    try:
        conn = mysql.connector.connect(host="localhost", user="root", password="123456", database="face")
        if conn.is_connected():
            print("Kết nối MySQL thành công!")
        return conn
    except Error as e:
        print(f"Không thể kết nối MySQL: {e}")
        return None

mysql_conn = check_mysql_connection()

# Extract faces from an image
def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faceSamples = []
    ids = []

    for imagePath in imagePaths:
        PIL_img = Image.open(imagePath).convert('L')
        img_numpy = np.array(PIL_img, 'uint8')

        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = face_detector.detectMultiScale(img_numpy)

        for (x, y, w, h) in faces:
            faceSamples.append(img_numpy[y:y + h, x:x + w])
            ids.append(id)

    return faceSamples, ids

def train(namesarr, newusername, newuserid):
    cam = cv2.VideoCapture(0)  # Mở lại camera
    if not cam.isOpened():
        print("\n [ERROR] Không thể mở camera.")
        return

    cam.set(3, 640)  # Chiều rộng khung hình
    cam.set(4, 480)  # Chiều cao khung hình


    # Nhập ID khuôn mặt
    Name_id = f"{newusername}_{newuserid}"
    frame_skip = 3
    frame_counter = 0
    namearr = namesarr
    namearr.append(Name_id)
    face_id = len(namearr) - 1
    writenames(namearr)
    print("\n [INFO] Khởi tạo camera...")

    count = 0
    font = cv2.FONT_HERSHEY_SIMPLEX
    while True:
        ret, img = cam.read()
        if not ret or img is None or img.size == 0:
            print("\n [ERROR] Không thể lấy được khung hình từ camera.")
            break

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            if frame_counter % frame_skip == 0:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                count += 1
                cv2.putText(img, f'{count}/50', (x + 5, y + h - 5), font, 1, (255, 255, 0), 2)
                cv2.imwrite(f"dataset/User.{face_id}.{count}.jpg", gray[y:y + h, x:x + w])
            frame_counter  +=1

        cv2.imshow('Nhan dien khuon mat', img)
        k = cv2.waitKey(100) & 0xff
        if k == 27 or count >= 50:  # Lấy 50 ảnh và thoát
            break

    print("\n [INFO] Đã lấy được 50 ảnh.")
    cam.release()
    cv2.destroyAllWindows()

    print("\n [INFO] Đang training dữ liệu ...")
    faces, ids = getImagesAndLabels('dataset')
    recognizer.train(faces, np.array(ids))
    recognizer.write('trainer/trainer.yml')
    print(f"\n [INFO] {len(np.unique(ids))} khuôn mặt đã được train.")


def facereco():
    try:
        recognizer.read('trainer/trainer.yml')
    except cv2.error as e:
        print("\n [ERROR] Lỗi khi đọc mô hình: ", e)
        return

    faceCascade = cv2.CascadeClassifier(cascade_path)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cam = cv2.VideoCapture(0)  # Mở lại camera
    if not cam.isOpened():
        print("\n [ERROR] Không thể mở camera.")
        return

    cam.set(3, 640)
    cam.set(4, 480)
    minW = 0.1 * cam.get(3)
    minH = 0.1 * cam.get(4)

    print("\n [INFO] Nhấn ESC để dừng.")
    while True:
        ret, img = cam.read()
        if not ret or img is None or img.size == 0:
            print("\n [ERROR] Không thể lấy được khung hình từ camera.")
            break

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(int(minW), int(minH)))

        for (x, y, w, h) in faces:
            try:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                id, confidence = recognizer.predict(gray[y:y + h, x:x + w])

                if confidence < 70:
                    id = namearr[id]
                    confidence = f"{round(100 - confidence)}%"
                else:
                    id = "unknown"
                    confidence = f"{round(100 - confidence)}%"

                cv2.putText(img, str(id), (x + 5, y - 5), font, 2, (255, 0, 200), 2)
                add_attendance(id)
                cv2.putText(img, str(confidence), (x + 5, y + h - 5), font, 1, (255, 255, 0),2 )
            except ValueError as e:
                print("\n [ERROR] Lỗi trong quá trình dự đoán: ", e)
                continue

        cv2.imshow('Nhan dien khuon mat', img)

        k = cv2.waitKey(10) & 0xff  # Nhấn ESC để thoát video
        if k == 27:
            break

    print("\n [INFO] Thoát.")
    cam.release()
    cv2.destroyAllWindows()


# Extract today's attendance from CSV file
def extract_attendance():
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    names = df['Name']
    rolls = df['Roll']
    times = df['Time']
    l = len(df)
    return names, rolls, times, l

# Add attendance for a specific user
def add_attendance(name):
    username, userid = name.split('_')
    current_time = datetime.now().strftime("%H:%M:%S")

    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    if int(userid) not in df['Roll'].astype(int).tolist():
        with open(f'Attendance/Attendance-{datetoday}.csv', 'a') as f:
            f.write(f'\n{username},{userid},{current_time}')

################## ROUTING FUNCTIONS #######################

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/Login', methods=['GET', 'POST'])
def Login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        print(f"Username: {username}, Password: {password}") # Debugging

    try:
        # Kết nối đến cơ sở dữ liệu MySQL
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="123456",
            database="face"
        )
        cursor = conn.cursor()

        # Lấy thông tin người dùng theo email
        cursor.execute("SELECT * FROM loginn WHERE username = %s AND pwd = %s", (username,password))
        user = cursor.fetchone()
        conn.close()

        if user:
            stored_password = user[2]  # Điều chỉnh chỉ số dựa trên cấu trúc bảng của bạn
            if password == stored_password:  # So sánh mật khẩu rõ ràng
                session['user'] = user[0]  # Điều chỉnh chỉ số nếu cần thiết
                return redirect(url_for('home'))
            else:
                flash("Sai tên hoặc mật khẩu!", "error")
        else:
            flash("Sai tên hoặc mật khẩu", "error")

    except mysql.connector.Error as err:
        flash(f"Error: {err}", "error")

    return redirect(url_for('index'))

#return render_template('home.html')  # Render form khi GET

# Route xử lý đăng ký
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        email = request.form.get('email')

        if not username or not password or not email:
            flash("Vui lòng điền đầy đủ thông tin.", "error")
            return redirect(url_for('index'))

        # Kết nối với cơ sở dữ liệu MySQL
        conn = check_mysql_connection()
        if conn is None:
            flash("Không thể kết nối với cơ sở dữ liệu. Vui lòng thử lại sau.", "error")
            return redirect(url_for('index'))

        cursor = conn.cursor()

        # Kiểm tra nếu tên đăng nhập đã tồn tại
        cursor.execute("SELECT * FROM loginn WHERE username=%s", (username,))
        existing_user = cursor.fetchone()

        if existing_user:
            flash("Tên đăng nhập này đã tồn tại, vui lòng sử dụng tên khác.", "error")
            conn.close()
            return redirect(url_for('index'))

        # Kiểm tra nếu email đã tồn tại
        cursor.execute("SELECT * FROM loginn WHERE email=%s", (email,))
        existing_email = cursor.fetchone()

        if existing_email:
            flash("Email này đã tồn tại, vui lòng sử dụng email khác.", "error")
            conn.close()
            return redirect(url_for('index'))

        # Thêm người dùng mới vào cơ sở dữ liệu
        query = "INSERT INTO loginn (username, email, pwd) VALUES (%s, %s, %s)"
        cursor.execute(query, (username, email, password))
        conn.commit()
        conn.close()

        flash("Đăng ký thành công! Bạn có thể đăng nhập bây giờ.", "success")
        return redirect(url_for('index'))
@app.route('/home')
def home():
    if 'user' not in session:
        return redirect(url_for('index'))
    names, rolls, times, l = extract_attendance()
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return render_template('home.html',
                           names=names,
                           rolls=rolls,
                           times=times,
                           l=l,
                           totalreg=len(namearr),
                           current_time=current_time,
                           logged_in=True,
                           page='home')
"""
@app.route('/dashboard')
def dashboard():
    if 'user' not in session:
        return redirect(url_for('home'))
    return "Chào mừng đến với ứng dụng chính!"
"""
@app.route('/start', methods=['GET'])
def start():
    if 'trainer.yml' not in os.listdir('trainer'):
        return render_template('home.html',
                               totalreg=len(namearr),
                               mess='There is no trained model in the static folder. Please add a new face to continue.',
                               logged_in='username' in session,
                               page='home')
    facereco()

    names, rolls, times, l = extract_attendance()
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return render_template('home.html',
                           names=names,
                           rolls=rolls,
                           times=times,
                           l=l,
                           totalreg=len(namearr),
                           current_time=current_time,
                           logged_in='username' in session,
                           page='home')

    # Chức năng thêm người dùng mới
@app.route('/add', methods=['GET', 'POST'])
def add():
    if request.method == 'POST':
        username = request.form['newusername']
        userid = request.form['newuserid']

        # Lưu khuôn mặt vào thư mục

        # Training và lưu trữ khuôn mặt mới
        train(namearr, username, userid)

        # Chuyển về trang chủ
        return redirect(url_for('home'))

    return render_template('home.html', logged_in='username' in session)


    names, rolls, times, l = extract_attendance()
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return render_template('home.html',
                            names=names,
                               rolls=rolls,
                               times=times,
                               l=l,
                               totalreg=len(namearr),
                               current_time=current_time,
                               logged_in='username' in session,
                               page='home')

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('index'))

@app.route('/attendance_sheet')
def attendance_sheet():
    names, rolls, times, l = extract_attendance()
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return render_template('home.html',
                           names=names,
                           rolls=rolls,
                           times=times,
                           l=l,
                           totalreg=len(namearr),
                           current_time=current_time,
                           logged_in='username' in session,
                           page='attendance_sheet')


# Main function to run Flask App
if __name__ == '__main__':
    app.run(debug=True, port=5003)
