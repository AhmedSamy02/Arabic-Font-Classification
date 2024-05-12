from flask import Flask, request, jsonify
import numpy as np
import pickle
from correct_orientation import *
from crop_image import *
from lpq import *
fonts = ['Marhey','Lemonada','Scheherazade','IBM']
app = Flask(__name__)

model = pickle.load(open('random_99.pkl', 'rb'))
lpq = LPQ(5)

@app.route('/predict', methods=['POST'])
def predict():
    # Check if request contains an image
    # if 'image' not in request.files:
    #     return jsonify({'error': 'No image provided'})
    # Read the image file
    # image = request.files['image']
    try:
        image =request.files['image'].stream.read()
        
        image = cv2.imdecode(np.fromstring(image, np.uint8), cv2.IMREAD_COLOR)
        count =[0,0,0,0]
        processed_image = preprocess_image(image)
        
        cropped_images = crop_image(processed_image,4,True)
        predictions=[]
        for cropped in cropped_images:
            
            prediction = model.predict([lpq.__call__(cropped)])
            predictions.append(prediction)
        count[0] =predictions.count(0)
        count[1] =predictions.count(1)
        count[2] =predictions.count(2)
        count[3] =predictions.count(3)
        final_predection = np.argmax(count)
        
        return jsonify({'label': fonts[final_predection],})
    except Exception as e:
        return jsonify({'error': str(e)})
if __name__ == '__main__':
    app.run(debug=False)
