import cv2
import numpy as np
from tflite_class import TfLiteModel

# define , load models
face_detect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
mask_model_path = "mask_model.tflite"
a_g_model_path = "age_gender_model.tflite"
not_found = cv2.imread('notfound.jpg')

mask_model = TfLiteModel(mask_model_path)
a_g_model = TfLiteModel(a_g_model_path)


class GenerateVideo(object):
    def __init__(self):
        self.video = cv2.VideoCapture(1)

    def __del__(self):
        self.video.release()

    # returns camera frames along with bounding boxes and predictions
    def get_frame(self):
        _, img = self.video.read()
        # face detection
        img = img.to_ndarray(format="bgr24")
        faces = face_detect.detectMultiScale(img, scaleFactor=1.1, minNeighbors=4)

        for (x, y, w, h) in faces:
            # face images processing
            face_img = img[y:y + h, x:x + w]
            face_img1 = input_process(face_img)

            # predict mask / no mask

            mask_pred = mask_model.model_predict(face_img1)

            if mask_pred > 0:  # no mask
                face_img2 = input_process(face_img, shape=(64, 64))
                gender_pred, age_pred = a_g_model.model_predict(face_img2)
                if gender_pred[0][0] > 0.5:
                    gender = 'Female'
                else:
                    gender = 'Male'
                ages = np.arange(0, 101).reshape(101, 1)
                age_pred = age_pred.dot(ages).flatten()
                mask_pred = 'No Mask'
                color = (0, 0, 255)
                text = mask_pred + '  ' + gender + '  ' + str(int(age_pred))
            else:
                text = 'MASK'
                color = (0, 255, 0)
            cv2.putText(img, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            # _, jpeg = cv2.imencode('.jpg', img)
        _, jpeg = cv2.imencode(".jpg", img)
        return jpeg.tobytes()

    def get_image(self):
        _, img = self.video.read()
        _, jpeg = cv2.imencode(".jpg", img)
        return jpeg.tobytes()
#<camera data-app-id='a-a14a2530-fb06-0139-4de0-0aac5b511429' id='myCamera'></camera>
#               <img id="bg" src="{{ url_for('video_feed') }}" style="width: 800px;">
def input_process(image, shape=(224, 224)):
    out_image = cv2.resize(image, shape)
    out_image = out_image[np.newaxis]
    out_image = np.array(out_image, dtype=np.float32)
    return out_image


#<script src='//cameratag.com/v14/js/cameratag.js' type='text/javascript'></script>
#<link rel='stylesheet' href='//cameratag.com/static/14/cameratag.css'>