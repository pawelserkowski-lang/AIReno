import cv2
import numpy as np
import os

class ImageProcessor:
    @staticmethod
    def extract_with_strategy(image_path, output_folder, strategy=0, min_area_percent=0.06):
        strategies = [
            ImageProcessor._strategy_morphology,
            ImageProcessor._strategy_otsu,
            ImageProcessor._strategy_adaptive
        ]
        method = strategies[strategy % len(strategies)]
        return method(image_path, output_folder, min_area_percent)

    @staticmethod
    def _base_processing(image_path, output_folder, edge_image, min_area_percent):
        img = cv2.imread(image_path)
        original = img.copy()
        h, w = img.shape[:2]
        min_area = (h * w) * min_area_percent
        
        contours, _ = cv2.findContours(edge_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        extracted = []
        count = 0
        
        for c in sorted(contours, key=cv2.contourArea, reverse=True):
            area = cv2.contourArea(c)
            if area < min_area: continue
            if area > (h*w)*0.98: continue 

            x, y, cw, ch = cv2.boundingRect(c)
            ar = float(cw) / ch
            if ar < 0.25 or ar > 4.0: continue 

            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            
            if len(approx) == 4:
                try:
                    warped = ImageProcessor.four_point_transform(original, approx.reshape(4, 2))
                    wh, ww = warped.shape[:2]
                    mh, mw = int(wh*0.01), int(ww*0.01)
                    if mh>0 and mw>0: cropped = warped[mh:wh-mh, mw:ww-mw]
                    else: cropped = warped
                    
                    filename = f"crop_{count}_s{np.random.randint(99)}_{os.path.basename(image_path)}"
                    path = os.path.join(output_folder, filename)
                    cv2.imwrite(path, cropped)
                    extracted.append(path)
                    count += 1
                except: pass
        return extracted

    @staticmethod
    def _strategy_morphology(image_path, output_folder, min_area_percent):
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.medianBlur(gray, 9)
        edged = cv2.Canny(blurred, 30, 150)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel, iterations=4)
        return ImageProcessor._base_processing(image_path, output_folder, closed, min_area_percent)

    @staticmethod
    def _strategy_otsu(image_path, output_folder, min_area_percent):
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return ImageProcessor._base_processing(image_path, output_folder, thresh, min_area_percent)

    @staticmethod
    def _strategy_adaptive(image_path, output_folder, min_area_percent):
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        dilated = cv2.dilate(thresh, kernel, iterations=2)
        return ImageProcessor._base_processing(image_path, output_folder, dilated, min_area_percent)

    @staticmethod
    def four_point_transform(image, pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0], rect[2] = pts[np.argmin(s)], pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1], rect[3] = pts[np.argmin(diff)], pts[np.argmax(diff)]
        (tl, tr, br, bl) = rect
        width = max(int(np.linalg.norm(br - bl)), int(np.linalg.norm(tr - tl)))
        height = max(int(np.linalg.norm(tr - br)), int(np.linalg.norm(tl - bl)))
        dst = np.array([[0,0],[width-1,0],[width-1,height-1],[0,height-1]], dtype="float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        return cv2.warpPerspective(image, M, (width, height))
