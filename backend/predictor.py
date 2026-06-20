import os
import cv2
from tensorflow.keras.models import load_model

model_test = load_model(
    "/mnt/c/Users/Praneeth Tadi/Documents/Coding/Machine Learning/ML Projects/Dog-cat-classify/dog-cat-classify/model/Dogcat-classifier-v4.keras"
)


def predict_image(image):
    image_ = cv2.imread(image)

    if image_ is None or image_.size == 0:
        raise ValueError("The uploaded file is not a valid image.")

    image_ = cv2.cvtColor(image_, cv2.COLOR_BGR2RGB)

    resize_img = cv2.resize(image_, (256, 256))
    resize_img = resize_img.reshape((1, 256, 256, 3))
    resize_img = resize_img / 255.0

    pred = model_test.predict(resize_img, verbose=0)

    score = float(pred[0][0])

    if score <= 0.5:
        ver = "cat"
    else:
        ver = "dog"

    return ver, score