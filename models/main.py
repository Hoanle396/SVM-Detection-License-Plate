import cv2
import numpy as np
from .lib_detection import load_model, detect_lp, im2single

def sort_contours(cnts):
    reverse = False
    i = 0
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    sorted_contours = [cnt for _, cnt in sorted(zip(boundingBoxes, cnts), key=lambda b: (b[0][1] >= 60, b[0][i]), reverse=reverse)]
    return sorted_contours

def detect(Ivehicle):
    wpod_net_path = "models/wpod-net.json"
    wpod_net = load_model(wpod_net_path)

    Dmax = 608
    Dmin = 288

    ratio = float(max(Ivehicle.shape[:2])) / min(Ivehicle.shape[:2])
    side = int(ratio * Dmin)
    bound_dim = min(side, Dmax)

    _, LpImg, lp_type = detect_lp(
        wpod_net, im2single(Ivehicle), bound_dim, lp_threshold=0.5
    )

    digit_w = 30
    digit_h = 60

    model_svm = cv2.ml.SVM_load("models/svm.xml")

    if len(LpImg):
        LpImg[0] = cv2.convertScaleAbs(LpImg[0], alpha=(255.0))

        roi = LpImg[0]

        gray = cv2.cvtColor(LpImg[0], cv2.COLOR_BGR2GRAY)

        binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)[1]

        kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thre_mor = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel3)
        cont, _ = cv2.findContours(thre_mor, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        plate_info = ""

        for c in sort_contours(cont):
            (x, y, w, h) = cv2.boundingRect(c)
            ratio = h / w
            if 1.5 <= ratio <= 5 and h >= 60:
                if h / roi.shape[0] >= 0.6 if lp_type == 1 else 0.8: 
                    cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    curr_num = thre_mor[y : y + h, x : x + w]
                    curr_num = cv2.resize(curr_num, dsize=(digit_w, digit_h))
                    _, curr_num = cv2.threshold(curr_num, 30, 255, cv2.THRESH_BINARY)
                    curr_num = np.array(curr_num, dtype=np.float32)
                    curr_num = curr_num.reshape(-1, digit_w * digit_h)

                    result = model_svm.predict(curr_num)[1]
                    result = int(result[0, 0])

                    if result <= 9:
                        result = str(result)
                    else:
                        result = chr(result)

                    plate_info += result

    return plate_info if plate_info and len(plate_info) > 7 else None
