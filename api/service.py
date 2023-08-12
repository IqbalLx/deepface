import base64
import os
import time
from deepface import DeepFace

def base64_to_image(base64_string, output_dir, image_extension):
    # Decode the base64 string into bytes
    image_data = base64.b64decode(base64_string)

    # Create a unique filename using a timestamp
    timestamp = str(int(time.time()))
    image_filename = f"image_{timestamp}.{image_extension}"

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Construct the full path to save the image
    image_path = os.path.join(output_dir, image_filename)

    # Write the image data to the file
    with open(image_path, "wb") as image_file:
        image_file.write(image_data)

    return image_path

def represent(img_path, model_name, detector_backend, enforce_detection, align):
    result = {}
    embedding_objs = DeepFace.represent(
        img_path=img_path,
        model_name=model_name,
        detector_backend=detector_backend,
        enforce_detection=enforce_detection,
        align=align,
    )
    result["results"] = embedding_objs
    return result


def verify(
    img1_path, img2_path, model_name, detector_backend, distance_metric, enforce_detection, align
):
    obj = DeepFace.verify(
        img1_path=img1_path,
        img2_path=img2_path,
        model_name=model_name,
        detector_backend=detector_backend,
        distance_metric=distance_metric,
        align=align,
        enforce_detection=enforce_detection,
    )
    return obj


def analyze(img_path, actions, detector_backend, enforce_detection, align):
    result = {}
    demographies = DeepFace.analyze(
        img_path=img_path,
        actions=actions,
        detector_backend=detector_backend,
        enforce_detection=enforce_detection,
        align=align,
    )
    result["results"] = demographies
    return result
