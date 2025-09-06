from flask import Flask, jsonify, request
import numpy as np
from flask_cors import CORS

import json
from src.nnetwork import Network
from src.activation_function import LeakyReLU

app = Flask(__name__)

CORS(app)

nn = Network( layer_sizes=None, activation_func=LeakyReLU(), weight_file='models/current_model/weights.npz', bias_file='models/current_model/biases.npz')


@app.route('/api/predict', methods=['POST'])
def predict():
    '''
    Takes 2
    '''
    try:
        data = request.get_json()

        if not data or 'array' not in data:
            return jsonify({
                'error': 'Missing required field: array',
                'message': 'Please provide a 2D array in the request body',
                'example': {'array': [[1, 2, 3], [4, 5, 6]]}
            }), 400
        
        array_2d = np.array(data['array'])

        if array_2d.ndim != 2:
            return jsonify({
            'error': 'Invalid array dimensions',
            'message': f'Expected 2D array, got {array_2d.ndim}D array',
            'shape': array_2d.shape
        }), 400
        
        total_mass = np.sum(array_2d)
        if total_mass == 0:
            # Handle edge case of all zeros
            center_of_mass_y = array_2d.shape[0] / 2
            center_of_mass_x = array_2d.shape[1] / 2
        else:
            # Calculate center of mass coordinates
            y_coords, x_coords = np.mgrid[0:array_2d.shape[0], 0:array_2d.shape[1]]
            center_of_mass_y = float(np.sum(y_coords * array_2d) / total_mass)
            center_of_mass_x = float(np.sum(x_coords * array_2d) / total_mass)
        
        # Calculate shifts to center the image at position (14, 14)
        shift_y = int(np.round(14 - center_of_mass_y))
        shift_x = int(np.round(14 - center_of_mass_x))
        
        # Center the array by rolling it
        centered_array = np.roll(array_2d, shift=shift_y, axis=0)
        centered_array = np.roll(centered_array, shift=shift_x, axis=1)
        
        # Center the array by rolling it
        centered_array = np.roll(array_2d, shift=shift_y, axis=0)
        centered_array = np.roll(centered_array, shift=shift_x, axis=1)


        flattened_array = centered_array.flatten()

        probs = nn.calculate_single_output(flattened_array)
        print(len(probs))

        predicted_class = np.argmax(probs)
        confidence = np.max(probs)

        for i, prob in enumerate(probs):
            print(f"Class {i}: {prob:.3f}")
        
        # Return results
        return jsonify({
            'status': 'success',
            'flattened_length': len(flattened_array),
            'probabilities' : json.dumps(probs.tolist()),
            'predicted_class': int(predicted_class),
            'confidence': float(confidence)
        })
        
    except ValueError as e:
        return jsonify({
            'error': 'Invalid array data',
            'message': f'Could not process array: {str(e)}'
        }), 400
    except Exception as e:
        return jsonify({
            'error': 'Processing error',
            'message': str(e)
        }), 400


# For Vercel deployment
app_handler = app

if __name__ == '__main__':
    # For local development
    app.run(debug=True, host='0.0.0.0', port=5000)