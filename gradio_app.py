'''A script to detect skin defects'''
import cv2
import numpy as np
import mediapipe as mp
import gradio as gr

def convert_bgr2rgb(image):
    '''Convert an image from BGR to RGB color space.'''
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def detect_faces(colored_img):
    '''
    Detect faces in an image using MediaPipe and draw bounding boxes around them.
    '''
    img_copy = np.copy(colored_img)
    print("Detecting faces using MediaPipe...")
    face_detection = mp.solutions.face_detection.FaceDetection(
        model_selection=1,
        min_detection_confidence=0.5
        )
    results = face_detection.process(cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB))
    if not results.detections:
        print("No faces detected.")
        return img_copy, []
    print(f"Number of faces detected: {len(results.detections)}")
    cropped_faces = []
    for detection in results.detections:
        bbox_c = detection.location_data.relative_bounding_box
        ih, iw, _ = img_copy.shape
        x, y = int(bbox_c.xmin * iw), int(bbox_c.ymin * ih)
        w, h = int(bbox_c.width * iw), int(bbox_c.height * ih)
        x = max(0, x)
        y = max(0, y)
        w = min(iw - x, w)
        h = min(ih - y, h)
        cv2.rectangle(img_copy, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cropped_face = img_copy[y:y+h, x:x+w]
        cropped_faces.append(cropped_face)
    return img_copy, cropped_faces

def enhance_contrast(image):
    '''Enhance the contrast of an image using histogram equalization and CLAHE.'''
    print("Applying HDR effect...")
    if len(image.shape) != 3 or image.shape[2] != 3:
        print("Error: Input image is not in the correct format. Expected a BGR image.")
        return image
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.equalizeHist(l)
    scale_factor = image.shape[0] / 1024
    clahe = cv2.createCLAHE(
        clipLimit=1.7 * scale_factor,
        tileGridSize=(int(10 * scale_factor), int(10 * scale_factor))
        )
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    enhanced_image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    enhanced_image = cv2.convertScaleAbs(enhanced_image, alpha=1.3, beta=-50)
    return enhanced_image

def highlight_skin_issues(face_img):
    '''Highlight potential skin issues such as redness on a face'''
    print("Highlighting skin issues...")
    enhanced_img = enhance_contrast(face_img)
    lab_img = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab_img)
    lower_a = np.array([147])
    upper_a = np.array([160])
    red_mask = cv2.inRange(a_channel, lower_a, upper_a)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)
    results = face_mesh.process(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
    if results.multi_face_landmarks:
        mask = np.zeros(face_img.shape[:2], dtype="uint8")
        for face_landmarks in results.multi_face_landmarks:
            points = np.array([(int(pt.x * face_img.shape[1]), int(pt.y * face_img.shape[0])) for pt in face_landmarks.landmark])
            face_contour_indices = [
                10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
            ]
            face_contour = points[face_contour_indices]
            cv2.drawContours(mask, [face_contour], -1, 255, thickness=cv2.FILLED)
            # Exclude
            mesh_annotations = {
                'lips': [
                    61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 146, 91, 181, 84, 17,
                    314, 405, 321, 375, 291, 78, 191, 80, 81, 82, 13, 312, 311, 310,
                    415, 308, 95, 88, 178, 87, 14, 317, 402, 318, 324
                ],
                'leftEye': [
                    263, 249, 390, 373, 374, 380, 381, 382, 362, 466, 388, 387, 386,
                    385, 384, 398, 467, 260, 259, 257, 258, 286, 414, 359, 255, 339,
                    254, 253, 252, 256, 341, 463, 342, 445, 444, 443, 442, 441, 413,
                    446, 261, 448, 449, 450, 451, 452, 453, 464, 372, 340, 346, 347,
                    348, 349, 350, 357, 465
                ],
                'rightEye': [
                    33, 7, 163, 144, 145, 153, 154, 155, 133, 246, 161, 160, 159, 158,
                    157, 173, 247, 30, 29, 27, 28, 56, 190, 130, 25, 110, 24, 23, 22,
                    26, 112, 243, 113, 225, 224, 223, 222, 221, 189, 226, 31, 228, 229,
                    230, 231, 232, 233, 244, 143, 111, 117, 118, 119, 120, 121, 128, 245
                ],
                'leftEyebrow': [
                    383, 300, 293, 334, 296, 336, 285, 417,
                    265, 353, 276, 283, 282, 295
                ],
                'rightEyebrow': [
                    35, 124, 46, 53, 52, 65, 156, 70, 63, 105, 66, 107, 55, 193
                ]
            }     
            for key in mesh_annotations:
                idx_list = mesh_annotations[key]
                region_points = points[idx_list]
                if len(region_points) > 0:
                    region_contour = np.array(region_points, dtype=np.int32)
                    cv2.drawContours(mask, [region_contour], -1, 0, thickness=cv2.FILLED)
            skin_mask = cv2.bitwise_and(red_mask, red_mask, mask=mask)
    else:
        print("No face landmarks detected.")
        return face_img
    contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"Number of potential skin issues detected: {len(contours)}")
    min_area = 1e-5 * face_img.shape[0] * face_img.shape[1]
    max_area = 1e-3 * face_img.shape[0] * face_img.shape[1]
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area < area < max_area:
            overlay = face_img.copy()
            cv2.drawContours(overlay, [contour], -1, (0, 0, 255), thickness=cv2.FILLED)
            alpha = 0.5
            face_img = cv2.addWeighted(overlay, alpha, face_img, 1 - alpha, 0)
    print("Skin issues highlighted successfully.")
    return face_img

def process_image(image_path):
    '''Process an input image by detecting faces and highlighting potential skin issues.'''
    img = cv2.imread(image_path)
    if img is None:
        return None
    img_with_faces, cropped_faces = detect_faces(img)
    processed_faces = []
    for idx, face in enumerate(cropped_faces):
        print(f"Processing face {idx+1}...")
        processed_face = highlight_skin_issues(face)
        processed_faces.append(processed_face)
    print("Image processing completed.")
    return convert_bgr2rgb(processed_faces[0])

iface = gr.Interface(
    fn=process_image,
    inputs=gr.Image(type="filepath"),
    outputs="image",
    title="Skin Defect Detection",
    description="Upload an image, and the app will detect and highlight skin defects."
)

if __name__ == "__main__":
    iface.launch()
