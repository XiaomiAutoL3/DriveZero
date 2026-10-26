"""Decode packed camera JPEGs at the student's input resolution."""
import os
import numpy as np

CAMERA_IMAGE_WIDTH = 960
CAMERA_IMAGE_HEIGHT = 512
_JPEG_DECODER = None
_JPEG_DECODER_PID = None

def resize_camera_image(image: np.ndarray) -> np.ndarray:
    if image.shape[:2] != (CAMERA_IMAGE_HEIGHT, CAMERA_IMAGE_WIDTH):
        import cv2

        image = cv2.resize(
            image,
            (CAMERA_IMAGE_WIDTH, CAMERA_IMAGE_HEIGHT),
            interpolation=cv2.INTER_LINEAR,
        )
    return image.astype(np.uint8, copy=False)


def get_jpeg_decoder():
    global _JPEG_DECODER, _JPEG_DECODER_PID
    pid = os.getpid()
    if _JPEG_DECODER is None or _JPEG_DECODER_PID != pid:
        from turbojpeg import TurboJPEG

        _JPEG_DECODER = TurboJPEG()
        _JPEG_DECODER_PID = pid
    return _JPEG_DECODER


def decode_camera_image(img_data: bytes) -> np.ndarray:
    from turbojpeg import TJPF_RGB

    jpeg = get_jpeg_decoder()
    width, height, _, _ = jpeg.decode_header(img_data)
    shorter_side = min(width, height)
    target_shorter_side = min(CAMERA_IMAGE_WIDTH, CAMERA_IMAGE_HEIGHT)
    if shorter_side >= target_shorter_side * 4:
        scale = (1, 4)
    elif shorter_side >= target_shorter_side * 2:
        scale = (1, 2)
    else:
        scale = (1, 1)
    return resize_camera_image(
        jpeg.decode(img_data, pixel_format=TJPF_RGB, scaling_factor=scale)
    )


