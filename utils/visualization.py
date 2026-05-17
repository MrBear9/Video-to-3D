
import numpy as np
import cv2


def draw_text_on_image(image: np.ndarray, text: str, position: tuple = (10, 30),
                       font_scale: float = 1.0, color: tuple = (0, 255, 0),
                       thickness: int = 2) -&gt; np.ndarray:
    img = image.copy()
    cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, color, thickness, cv2.LINE_AA)
    return img


def draw_bounding_box(image: np.ndarray, bbox: np.ndarray,
                      color: tuple = (0, 255, 0), thickness: int = 2) -&gt; np.ndarray:
    img = image.copy()
    x1, y1 = int(bbox[0][0]), int(bbox[0][1])
    x2, y2 = int(bbox[1][0]), int(bbox[1][1])
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    return img


def concatenate_images(images: list, axis: int = 1) -&gt; np.ndarray:
    return np.concatenate(images, axis=axis)


def resize_image(image: np.ndarray, size: tuple) -&gt; np.ndarray:
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)

