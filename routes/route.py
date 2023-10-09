from flask import jsonify, request
import cv2
from models.main import detect
import numpy as np
import datetime


def extractRoute(app):
    @app.post("/extract")
    def upload():
        try:
            if "image" not in request.files:
                return {"detail": "No file found"}, 400
            image_file = request.files["image"]
            image = cv2.imdecode(
                np.fromstring(image_file.read(), np.uint8), cv2.IMREAD_UNCHANGED
            )
            result = detect(image)
            if result != "":
                return {
                    "status": "OK",
                    "result": result,
                    "message": "Detected image successfully!",
                }, 200
            return {
                "status": "FAILED",
                "result": None,
                "message": "Detected image failed!",
            }, 400
        except Exception as e:
            print(e)
            return {
                "status": "FAILED",
                "result": None,
                "message": "Something went wrong!",
            }, 500
