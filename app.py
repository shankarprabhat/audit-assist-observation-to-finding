from flask import Flask, jsonify, request
from flask_cors import CORS  # Import Flask-CORS
import generate_audit_finding as af
import traceback

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/')
def hello():
    return "Hello, World!"

@app.route('/api/get-audit-findings', methods=['POST'])
def get_audit_findings():
    try:
        
        req_body = request.get_json()
        input_observation = req_body.get('auditObservation')
        example_string = af.prepare_data()  # Load the example data

        try:
            audit_finding, status_code = af.return_audit_findings(input_observation, example_string)
        except Exception as e:
            return jsonify(traceback.format_exception(e))
        
        # Convert DataFrame to JSON string
        json_data = audit_finding

        return jsonify(json_data)

    except Exception as e:
        return jsonify({'error': str(e)}), 500  # Return error with appropriate status code

if __name__ == '__main__':
    app.run(debug=True)  # Set debug=False for production