from flask import Flask, render_template, request
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input

app = Flask(__name__)

Part = {0: 'p01', 1: 'p02', 2: 'p03', 3: 'p04', 4: 'p05', 5: 'p06', 6: 'p07', 7: 'p08', 8: 'p09', 9: 'p10', 10: 'p11', 11: 'p12', 12: 'p13', 13: 'p14', 14: 'p15'}
damage = {0: 'Base', 1: 'Minor', 2: 'Moderate', 4: 'Severe'}

import sys
sys.path.append('/root/Web-Application-/templates/Part.h5')

from efficientnet.layers import Swish, DropConnect
from efficientnet.model import ConvKernalInitializer
from tensorflow.keras.utils import get_custom_objects

get_custom_objects().update({
    'ConvKernalInitializer': ConvKernalInitializer,
    'Swish': Swish,
    'DropConnect':DropConnect
})

model1 = tf.keras.models.load_model('/root/Web-Application-/templates/Part.h5')
model2 = tf.keras.models.load_model('/root/Web-Application-/templates/damage.h5')


model1.make_predict_function()
model2.make_predict_function()

# def predict_image1(img_path):
#     # Read the image and preprocess it
#     img = image.load_img(img_path, target_size=(150, 150))
#     x = image.img_to_array(img)
#     x = preprocess_input(x)
#     x = np.expand_dims(x, axis=0)
#     result = model1.predict(x)
#     return age[result.argmax()]

# def predict_image2(img_path):
#     # Read the image and preprocess it
#     img = image.load_img(img_path, target_size=(150, 150))
#     g = image.img_to_array(img)
#     g = preprocess_input(g)
#     g = np.expand_dims(g, axis=0)
#     result = model2.predict(g)
#     return gender[result.argmax()]
my_tuple = tuple(Part)

def predict_image1(img_path):
    # Read the image and preprocess it
    img = image.load_img(img_path, target_size=(150, 150))
    x = image.img_to_array(img)
    x = x.reshape((1,) + x.shape) 
    x /= 255.
    result = model1.predict(x)
    return my_tuple[int(result[0])]

def predict_image2(img_path):
    # Read the image and preprocess it
    img = image.load_img(img_path, target_size=(150, 150))
    g = image.img_to_array(img)
    g = g.reshape((1,) + g.shape) 
    g /= 255.
    result = model2.predict(g)
    return damage[result.argmax()]


# routes
@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Read the uploaded image and save it to a temporary file
        file = request.files['image']
        img_path = 'static/panoramic.jpg'
        file.save(img_path)

        # Predict the age

        Part_pred = predict_image1(img_path)
        damage_pred = predict_image2(img_path)

        # Render the prediction result
        return render_template('upload_completed.html', prediction1=Part_pred, prediction2=damage_pred)

if __name__ == '__main__':
    app.run(debug=True)