import io

import numpy as np
from flask import Flask, request, jsonify
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.models import Sequential, load_model
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

app = Flask(__name__)

trained_model = "verification_model.keras"


def create_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
    model.add(MaxPooling2D())
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


@app.route('/train', methods=['POST'])
def train_model():
    datagen = ImageDataGenerator(rescale=1.0 / 255.0, validation_split=0.2)

    train_gen = datagen.flow_from_directory(
        'images',
        class_mode='binary',
        batch_size=2,
        target_size=(150, 150),
        subset='training'
    )

    val_gen = datagen.flow_from_directory(
        'images',
        class_mode='binary',
        batch_size=2,
        target_size=(150, 150),
        subset='validation'
    )

    model = create_model()
    model.fit(train_gen, validation_data=val_gen, epochs=10)

    model.save(trained_model)

    return jsonify({'message': 'Model trained and saved'})


@app.route('/verify', methods=['POST'])
def verify_image():
    model = load_model(trained_model)

    img_file = request.files['image'].read()
    img = load_img(io.BytesIO(img_file), target_size=(150, 150))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    prediction = model.predict(img_array)

    if prediction[0][0] > 0.5:
        return jsonify({"message": "Local", "confidence": float(prediction[0][0])})
    else:
        return jsonify({"message": "Random", "confidence": float(1.0 - prediction[0][0])})


if __name__ == '__main__':
    app.run(debug=True)
