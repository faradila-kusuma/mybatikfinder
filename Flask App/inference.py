import matplotlib.image as mpimg
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from desc import batik_data

def read_class_names(file_path):
    with open(file_path, 'r') as file:
        lines = file.read().splitlines()
    return lines

# Reading text
class_names = read_class_names('Class_Names_New.txt')

def get_category(img_path):
    """Predict the category or label of a batik image using a trained TensorFlow model.

    Args:
        img_path (str): Path to the batik image file.

    Returns:
        dict: Dictionary containing predicted category and additional batik information.
    """

    # Image loading and preprocessing
    img = mpimg.imread(img_path)
    img = tf.cast(img, tf.float32)
    img = tf.image.resize(img, [224, 224])
    img = np.expand_dims(img, axis=0)

    # Load the TFLite model
    tflite_model_file = 'modelV2.tflite'
    #model_js = tfjs.converters.load_keras_model('D:\\Flask Testing\\tfjs_model')
    with open(tflite_model_file, 'rb') as fid:
        tflite_model = fid.read()

    # Create an interpreter and allocate tensors
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    # Get input and output tensor indices
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]

    # Make prediction
    prediction = []
    interpreter.set_tensor(input_index, img)
    interpreter.invoke()
    prediction.append(interpreter.get_tensor(output_index))

    # Get the predicted label
    predicted_label = np.argmax(prediction)
    predicted_index = np.argmax(prediction[0])
    class_names = read_class_names('Class_Names_New.txt')
    batik_description = batik_data[predicted_index]

    # Display the test image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.title('Batik Image')
    plt.xticks([])
    plt.yticks([])
    plt.show()

    # Display the predicted category and batik information
    print("Predicted category: {}".format(class_names[predicted_label]))
    
    print(f"    Asal Daerah: {batik_description['Asal Daerah']}")
    print(f"    Pola Batik: {batik_description['Pola Batik']}")
    print(f"    Teknik Pembuatan: {batik_description['Teknik Pembuatan']}")
    print(f"    Sejarah: {batik_description['Sejarah']}")

    return {
        "predicted_category": class_names[predicted_label],
        "origin": batik_description['Asal Daerah'],
        "pattern": batik_description['Pola Batik'],
        "technique": batik_description['Teknik Pembuatan'],
        "history": batik_description['Sejarah']
    }
