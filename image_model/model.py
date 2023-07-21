import tflite_runtime.interpreter as tflite
from PIL import Image
import numpy as np
import os
import sys
import time
import argparse

# Navigate to correct position in filesystem
script_directory = os.path.dirname(os.path.abspath(sys.argv[0]))
os.chdir(script_directory)

# Set up the model
def predict(model_path, image_path, labels_path):
  interpreter = tflite.Interpreter(model_path=model_path)
  interpreter.allocate_tensors()

  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()

  # Prepare and pass the input image  
  height = input_details[0]['shape'][1]
  width = input_details[0]['shape'][2]
  img = Image.open(image_path).resize((width, height))
  img = img.convert('RGB')
  input_data = np.array(img)
  input_data = np.expand_dims(img, axis=0)

  start_time = time.time()
  interpreter.set_tensor(input_details[0]['index'], input_data)

  # Make a prediction!
  interpreter.invoke()

  # Get and print the result
  output_data = interpreter.get_tensor(output_details[0]['index'])
  inf_time =  time.time() - start_time 
  print(f"time: {inf_time}s" )

  with open(labels_path, 'r') as f:
      labels = [line.strip() for line in f.readlines()]

  sorted_result = sorted((e,i) for i,e in enumerate(output_data[0]))
  prediction = sorted_result[-1:][::-1]
  return prediction, labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="To parse model path, image path and label path")
    parser.add_argument("model_path", type=str, help="Path to the model file.")
    parser.add_argument("image_path", type=str, help="Path to the image file.")
    parser.add_argument("labels_path", type=str, help="Path to the labels file.")
    args = parser.parse_args()

    if not args.model_path or not args.image_path or not args.labels_path:
        print("Warning: Must provide all model path, image path and label file path.")
        sys.exit(1)
    prediction, labels = predict(args.model_path, args.image_path, args.labels_path)
    j,i = prediction[0]
    print('{:08.6f}: {}'.format(float(j / 255.0), labels[i].split(":")[1]))
