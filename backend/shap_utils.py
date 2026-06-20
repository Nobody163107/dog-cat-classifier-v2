import shap
import cv2
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model
import uuid 
import os


model = load_model("/mnt/c/Users/Praneeth Tadi/Documents/Coding/Machine Learning/ML Projects/Dog-cat-classify/dog-cat-classify/model/Dogcat-classifier-v4.keras")


def processor(path): 
        img = cv2.imread(path)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = cv2.resize(img, (256, 256))

        # plt.imshow(img)

        # plt.show()

        img = img / 255.

        img_batch = np.expand_dims(img, axis = 0)
        return img_batch, img


def generate_shap(img_batch, img): 

    os.makedirs("static", exist_ok=True)
    masker = shap.maskers.Image("blur(8, 8)", img.shape)

    explainer = shap.Explainer(model.predict, masker)
    shap_val = explainer(img_batch, max_evals=200)
    plt.figure()
    shap.image_plot(shap_val,  show = False)
    filename = f"shap_{uuid.uuid4().hex}.png"
    
    output_path = f"static/{filename}"
    
    plt.savefig(
        output_path, bbox_inches = "tight"
    )
    plt.close()
    
    return filename