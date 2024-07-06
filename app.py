from flask import Flask, render_template, Response
import cv2
import tensorflow as tf
import numpy as np

app = Flask(__name__)

model = tf.keras.models.load_model('model/model.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/profile')
def profile():
    return render_template('profile.html')

@app.route('/detect')
def detect():
    return render_template('detect.html')

def generate_frames():
    camera = cv2.VideoCapture(0)
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            img = cv2.resize(frame, (224, 224))
            img = img / 255.0
            img = np.expand_dims(img, axis=0)
            prediction = model.predict(img)
            if prediction[0][0] > prediction[0][1]:
                label = 'Aluno'
            else:
                label = 'Outro'
            cv2.putText(frame, label, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
