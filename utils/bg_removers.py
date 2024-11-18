import onnxruntime as ort
import numpy as np
import cv2
import traceback
from PIL import Image
from pi_heif import register_heif_opener
import numpy as np
import cv2
import imutils

register_heif_opener()


model_path = "isnet_dis.onnx"
 
def normalize(image, mean, std):
    """Normalize a numpy image with mean and standard deviation."""
    return (image / 255.0 - mean) / std

def remove_background(model_path,im):
   
    input_size = (1024, 1024)

    try:
        # Load the ONNX model
        session = ort.InferenceSession(model_path)
                
        # If image is grayscale, convert to RGB
        if len(im.shape) == 2:
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
        
        # Normalize the image using NumPy
        im = im.astype(np.float32)  # Convert to float
        im_normalized = normalize(im, mean=[0.5, 0.5, 0.5], std=[1.0, 1.0, 1.0])
            
        # Resize the image
        im_resized = cv2.resize(im_normalized, input_size, interpolation=cv2.INTER_LINEAR)
        im_resized = np.transpose(im_resized, (2, 0, 1))  # CHW format
        im_resized = np.expand_dims(im_resized, axis=0)  # Add batch dimension

        # Run inference
        im_resized = im_resized.astype(np.float32)  
        ort_inputs = {session.get_inputs()[0].name: im_resized}
        ort_outs = session.run(None, ort_inputs)
            
        # Process the model output
        result = ort_outs[0][0]  # Assuming single output and single batch
        result = np.clip(result, 0, 1)  # Assuming you want to clip the result to [0, 1]
        result = (result * 255).astype(np.uint8)  # Rescale to [0, 255]
        result = np.transpose(result, (1, 2, 0))  # HWC format
        result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        # Resize to original shape
        original_shape = im.shape[:2]
        result = cv2.resize(result, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_LINEAR)

        # Ensure 'result' is 2D (H x W) and add an axis to make it (H x W x 1)
        alpha_channel = result[:, :, np.newaxis]

        # Concatenate the RGB channels of 'im' with the alpha channel
        im_rgba = np.concatenate((im, alpha_channel), axis=2)
        im_bgra = cv2.cvtColor(im_rgba, cv2.COLOR_RGBA2BGRA)

        
        # Define a solid black background with full opacity
        background_color = [0, 0, 0, 255]  # Black in BGRA order with full opacity for the alpha channel

        # Create a background image of the same size as our img with 4 channels (including alpha)
        background_image = np.full((im_rgba.shape[0], im_rgba.shape[1], 4), background_color, dtype=np.uint8)

        # Create a 3 channel alpha mask for blending, normalized to the range [0, 1]
        alpha_mask = (im_rgba[:, :, 3] / 255.0).reshape(im_rgba.shape[0], im_rgba.shape[1], 1)

        foreground = im_rgba[:, :, :3].astype(np.float32)  # Use only RGB channels of the foreground
        background = background_image[:, :, :3].astype(np.float32)  # Use only RGB channels of the background
        blended_image = (1 - alpha_mask) * background + alpha_mask * foreground
        blended_image = np.uint8(blended_image)

        # Convert from BGR to RGB for displaying in Matplotlib
        blended_image_rgb = cv2.cvtColor(blended_image, cv2.COLOR_BGR2RGB)
        

        return blended_image_rgb, result
    except Exception as e:
        print("An error occurred:")
        traceback.print_exc()
        return None,None
    
def refine_with_grabcut(image, approx_mask, iterations=5):
    """
    Refine receipt extraction using GrabCut algorithm with an approximate mask
    
    Args:
        image: RGB input image
        approx_mask: Binary mask where 255 is probable foreground (receipt)
        iterations: Number of GrabCut iterations
    
    Returns:
        Refined mask and extracted receipt
    """
    # Create mask for GrabCut
    # 0 = background, 1 = foreground, 2 = probable background, 3 = probable foreground
    mask = np.zeros(image.shape[:2], np.uint8)
    
    # Convert binary mask to GrabCut mask format
    mask[approx_mask == 0] = 0    # Definite background
    mask[approx_mask == 255] = 3  # Probable foreground
    
    # Create temporary arrays for GrabCut
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    
    # Run GrabCut
    try:
        cv2.grabCut(image, mask, None, bgd_model, fgd_model, iterations, cv2.GC_INIT_WITH_MASK)
    except Exception as e:
        print(f"GrabCut failed: {e}")
        return None, None
    
    # Create mask where foreground and probable foreground are white
    refined_mask = np.where((mask == 1) + (mask == 3), 255, 0).astype('uint8')
    
    # Apply refined mask to image
    refined_receipt = cv2.bitwise_and(image, image, mask=refined_mask)
    
    return refined_mask, refined_receipt

def extract_receipt_grabcut(img):
    """
    Complete pipeline for receipt extraction using GrabCut
    """
    # Load and preprocess image
  
    model_path = "isnet_dis.onnx"
    # Get initial mask using the original method
    
    _, initial_mask = remove_background(model_path,img)
    initial_mask = np.where(initial_mask > 0, 255, 0).astype('uint8')
    
    
    if initial_mask is None:
        print("Failed to create initial mask")
        return None
    
    # Refine using GrabCut
    refined_mask, refined_receipt = refine_with_grabcut(img, initial_mask)
    
    if refined_mask is None:
        print("GrabCut refinement failed")
        return None
    
    # Crop to content
    if np.sum(refined_mask) > 0:  # Check if we have any foreground
        # Find bounding box of the refined mask
        coords = cv2.findNonZero(refined_mask)
        x, y, w, h = cv2.boundingRect(coords)
        refined_receipt = refined_receipt[y:y+h, x:x+w]
    
    return refined_receipt


def load_image(image_path, target_width=500):
    """
    Load HEIC or regular image and preprocess it
    """
    register_heif_opener()
    im = Image.open(image_path)
    img = np.asarray(im)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = imutils.resize(img, width=target_width)
    return img

def extract_receipt(img):
    """
    Extract receipt area from image using color characteristics.
    Returns the cropped receipt and its mask.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Create binary mask for white/near-white regions
    # Using Otsu's thresholding to automatically determine optimal threshold
    blur = cv2.GaussianBlur(gray,(5,5),0)
    _, white_mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Clean up noise and small artifacts
    kernel = np.ones((5,5), np.uint8)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
    
    # Find contours in the mask
    contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None, None
    
    # Get the largest white area (assumed to be the receipt)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Create mask for the largest contour
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [largest_contour], -1, (255), -1)
    
    # Apply mask to original image
    receipt = cv2.bitwise_and(img, img, mask=mask)
    
    # Crop to bounding rectangle
    x, y, w, h = cv2.boundingRect(largest_contour)
    receipt_cropped = receipt[y:y+h, x:x+w]
    mask_cropped = mask[y:y+h, x:x+w]
    
    return receipt_cropped, mask_cropped

def process_receipt_image(image_path):
    """
    Main function to process receipt image
    """
    # Load and preprocess image
    img = load_image(image_path)
    
    # Extract receipt
    receipt, mask = extract_receipt(img)
    
    if receipt is None:
        print("No receipt detected in image")
        return None
    
    return receipt
