from flask import Flask, jsonify
import subprocess

app = Flask(__name__)

@app.route('/recognize-face', methods=['POST'])
def recognize():
    try:
        subprocess.run(["python", "your_face_script.py"])
        return jsonify({"success": True, "message": "Attendance marked"})
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})

if __name__ == '__main__':
    app.run(port=8000)