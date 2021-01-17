import os
from flask import Flask, request, render_template
import tensorflow as tf
from tensorflow import keras
import numpy as np

app = Flask(__name__)
UPLOAD_FOLDER ='static\\uploads'

model = tf.keras.models.load_model('model.h5')

class_names = ['Covid', 'Normal', 'Viral Pneumonia']
def model_predict(img_path):
	img = keras.preprocessing.image.load_img(
    	img_path, target_size=(150, 150)
		)
	img_array = keras.preprocessing.image.img_to_array(img)
	img_array = tf.expand_dims(img_array, 0) # Create a batch

	predictions = model.predict(img_array)
	score = tf.nn.softmax(predictions[0])

	return class_names[np.argmax(score)]



@app.route("/", methods=['GET', 'POST'])
def upload_predict():
	if request.method == "POST":
		image_file = request.files["image"]
		if image_file:
			image_location = os.path.join(
				UPLOAD_FOLDER,
				image_file.filename
				)
			image_file.save(image_location)
			return render_template("index.html", prediction=model_predict(image_location))

	return render_template("index.html", prediction="None")



if __name__ == "__main__":
	app.run(debug=True)