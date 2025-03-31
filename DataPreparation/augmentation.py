import cv2
import numpy as np

def apply_occlusion(image: np.ndarray, landmarks: list, size: int) -> np.ndarray:
    """
    Apply occlusion around specified landmarks.

    Parameters:
      image (np.ndarray): Grayscale image.
      landmarks (list): List of (x, y) tuples.
      size (int): Size of the square occlusion region.

    Returns:
      np.ndarray: Occluded image (uint8).
    """
    if not isinstance(size, int) or size < 0:
        raise ValueError("size must be a non-negative integer")

    occluded_image = image.copy()
    for x, y in landmarks:
        x_start = max(0, x - size)
        x_end = min(image.shape[1], x + size + 1)
        y_start = max(0, y - size)
        y_end = min(image.shape[0], y + size + 1)
        occluded_image[y_start:y_end, x_start:x_end] = 0
    return occluded_image.astype(np.uint8)

def apply_blur(image: np.ndarray, landmarks: list, size: int, severity: float) -> np.ndarray:
    """
    Apply Gaussian blur around specified landmarks.

    Parameters:
      image (np.ndarray): Grayscale image.
      landmarks (list): List of (x, y) tuples.
      size (int): Size of the square region.
      severity (float): Blur severity (0 to 1).

    Returns:
      np.ndarray: Blurred image (uint8).
    """
    if severity < 0:
        raise ValueError("severity must be non-negative")
    if not isinstance(size, int) or size < 0:
        raise ValueError("size must be a non-negative integer")

    blurred_image = image.astype(np.float32).copy()
    for x, y in landmarks:
        x_start = max(0, x - size)
        x_end = min(image.shape[1], x + size + 1)
        y_start = max(0, y - size)
        y_end = min(image.shape[0], y + size + 1)
        region = blurred_image[y_start:y_end, x_start:x_end]
        ksize = int(2 * severity + 1)  # Kernel size must be odd
        if ksize > 1:
            region = cv2.GaussianBlur(region, (ksize, ksize), 0)
        region = np.clip(region, 0, 255)
        blurred_image[y_start:y_end, x_start:x_end] = region
    return blurred_image.astype(np.uint8)

def gaussian_blur(image, severity):
    if severity < 1:
        return image  
    
    kernel_size = 2*severity + 1 # Define an odd kernel size based on severity
    blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    
    return blurred_image

def gaussian_noise(image, severity):
    if severity < 1:
        return image  

    noise = np.random.normal(0, severity, image.shape).astype(np.uint8)
    noisy_image = cv2.add(image, noise) 

    return noisy_image