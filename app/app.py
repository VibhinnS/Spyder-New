# app.py
from flask import Flask, render_template, request, jsonify
import json
import os

app = Flask(__name__)

# Sample data for available components
AVAILABLE_COMPONENTS = {
    "dies": [
        {"id": "cpu_die", "name": "CPU Die", "thickness": 0.5, "thermal_conductivity": 150, "type": "die"},
        {"id": "gpu_die", "name": "GPU Die", "thickness": 0.7, "thermal_conductivity": 130, "type": "die"},
        {"id": "memory_die", "name": "Memory Die", "thickness": 0.3, "thermal_conductivity": 100, "type": "die"}
    ],
    "materials": [
        {"id": "copper", "name": "Copper", "thermal_conductivity": 400, "type": "material"},
        {"id": "silicon", "name": "Silicon", "thermal_conductivity": 150, "type": "material"},
        {"id": "tim", "name": "Thermal Interface Material", "thermal_conductivity": 8.5, "type": "material"},
        {"id": "solder", "name": "Solder", "thermal_conductivity": 50, "type": "material"}
    ]
}


@app.route('/')
def index():
    return render_template('index.html', components=AVAILABLE_COMPONENTS)


@app.route('/save_stack', methods=['POST'])
def save_stack():
    stack_data = request.json
    with open('stack.json', 'w') as f:
        json.dump(stack_data, f, indent=2)
    

    return jsonify({
        "status": "success",
        "message": "Stack saved successfully",
        "stack": stack_data
    })

@app.route('/get_stack', methods=['GET'])
def get_stack():
    if not os.path.exists('stack.json'):
        return jsonify({"error": "No stack data found"}), 404

    with open('stack.json', 'r') as f:
        stack_data = json.load(f)

    return jsonify(stack_data)


@app.route('/analyze_stack', methods=['POST'])
def analyze_stack():
    stack_data = request.json
    # Here you would run your thermal analysis or optimization
    # For demonstration, we'll just return a placeholder result
    total_thickness = sum(layer.get('thickness', 0) for layer in stack_data)
    avg_thermal_conductivity = sum(layer.get('thermal_conductivity', 0) for layer in stack_data) / len(
        stack_data) if stack_data else 0

    results = {
        "total_thickness": total_thickness,
        "avg_thermal_conductivity": avg_thermal_conductivity,
        "thermal_resistance": total_thickness / avg_thermal_conductivity if avg_thermal_conductivity > 0 else "N/A",
        "stack_analysis": stack_data
    }

    print(results)
    return jsonify(results)


if __name__ == '__main__':
    app.run(debug=True, port=5189)