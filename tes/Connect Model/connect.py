import matplotlib.image as mpimg
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt



def read_class_names(file_path):
    with open(file_path, 'r') as file:
        lines = file.read().splitlines()
    return lines

# Gunakan fungsi untuk membaca class names dari file txt
class_names = read_class_names('class_names.txt')

# Cetak class names untuk memastikan pembacaan berfungsi
print("Class Names:", class_names)

def get_category(img_path):
    """Predict the category or label of an image using a trained TensorFlow model.

    Args:
        img_path (str): Path to the image file.

    Returns:
        str: Predicted category or label.
    """
    

    # Load the image and perform necessary preprocessing
    img = mpimg.imread(img_path)
    img = tf.cast(img, tf.float32)
    img = tf.image.resize(img, [224, 224])
    img = np.expand_dims(img, axis=0)

    # Load the TFLite model
    tflite_model_file = 'C:\\Tes\\model.tflite'
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
    class_names = read_class_names('C:\\Tes\\class_names.txt')

    # Cetak class_names untuk melihat nilai yang sesuai dengan dataset Anda


    # Display the test image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.title('Test Image')
    plt.xticks([])
    plt.yticks([])
    plt.show()

    # Display the predicted category
    print("Predicted category: {}".format(class_names[predicted_label]))

    return class_names[predicted_label]

# Example usage:
image_path = '30_jpg.rf.2a59a971c5535a8c2d4b14279b7ed1b8.jpg'
predicted_category = get_category(image_path)
