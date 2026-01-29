"""
Engineering Book Digitizer - Starter Template
A custom OCR solution for digitizing technical engineering books
with support for tables, subscripts, superscripts, and diagrams.
"""

import cv2
import numpy as np
import pytesseract
from pathlib import Path
import json
from datetime import datetime

pytesseract.pytesseract.tesseract_cmd = r'C:\Users\USER\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

class EngineeringBookOCR:
    def __init__(self, output_dir="output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def preprocess_image(self, image_path):
        """
        Preprocess the image for better OCR results
        Returns: preprocessed image as numpy array
        """
        print(f"Preprocessing: {image_path}")
        
        # Read image
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # ENHANCED: Remove shadows using morphological operations
        dilated = cv2.dilate(gray, np.ones((7,7), np.uint8))
        bg = cv2.medianBlur(dilated, 21)
        diff = 255 - cv2.absdiff(gray, bg)
        
        # Normalize to improve contrast
        normalized = cv2.normalize(diff, None, alpha=0, beta=255, 
                                   norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        # Denoise
        denoised = cv2.fastNlMeansDenoising(normalized, h=10)
        
        # Increase contrast using CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        contrasted = clahe.apply(denoised)
        
        # Sharpen the image
        kernel = np.array([[-1,-1,-1],
                          [-1, 9,-1],
                          [-1,-1,-1]])
        sharpened = cv2.filter2D(contrasted, -1, kernel)
        
        # Binarization (Otsu's method)
        _, binary = cv2.threshold(sharpened, 0, 255, 
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Save preprocessed image for inspection
        preprocessed_path = self.output_dir / f"preprocessed_{Path(image_path).name}"
        cv2.imwrite(str(preprocessed_path), binary)
        print(f"Saved preprocessed image to: {preprocessed_path}")
        
        return binary
    
    def detect_regions(self, image):
        """
        Detect different regions in the image (text, tables, figures)
        Returns: dictionary with region coordinates
        """
        print("Detecting regions...")
        
        # Simple region detection using contours
        # (This is basic - you'll enhance this later)
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        regions = {
            'text_blocks': [],
            'potential_tables': [],
            'potential_figures': []
        }
        
        height, width = image.shape
        min_area = (height * width) * 0.01  # Ignore very small regions
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
                
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 0
            
            region = {
                'x': x, 'y': y, 'width': w, 'height': h,
                'area': area, 'aspect_ratio': aspect_ratio
            }
            
            # Classify regions (basic heuristics)
            if aspect_ratio > 3:  # Wide regions might be tables
                regions['potential_tables'].append(region)
            elif aspect_ratio < 0.5:  # Tall thin regions might be figures
                regions['potential_figures'].append(region)
            else:
                regions['text_blocks'].append(region)
        
        print(f"Found {len(regions['text_blocks'])} text blocks, "
              f"{len(regions['potential_tables'])} potential tables, "
              f"{len(regions['potential_figures'])} potential figures")
        
        return regions
    
    def extract_text_basic(self, image, region=None):
        """
        Extract text from image or specific region
        Returns: extracted text
        """
        if region:
            x, y, w, h = region['x'], region['y'], region['width'], region['height']
            cropped = image[y:y+h, x:x+w]
        else:
            cropped = image
        
        # Use pytesseract for OCR
        custom_config = r'--oem 3 --psm 6'
        text = pytesseract.image_to_string(cropped, config=custom_config)
        
        return text
    
    def extract_text_with_boxes(self, image):
        """
        Extract text with bounding box information for sub/superscript detection
        Returns: list of text elements with positions
        """
        print("Extracting text with position data...")
        
        # Get detailed OCR data
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        
        text_elements = []
        n_boxes = len(data['text'])
        
        for i in range(n_boxes):
            if int(data['conf'][i]) > 30:  # Confidence threshold
                text = data['text'][i].strip()
                if text:
                    element = {
                        'text': text,
                        'x': data['left'][i],
                        'y': data['top'][i],
                        'width': data['width'][i],
                        'height': data['height'][i],
                        'confidence': data['conf'][i]
                    }
                    text_elements.append(element)
        
        print(f"Extracted {len(text_elements)} text elements")
        return text_elements
    
    def detect_formatting(self, text_elements):
        """
        Detect subscripts and superscripts based on vertical position
        Returns: formatted text with markup
        """
        if not text_elements:
            return ""
        
        # Calculate baseline (average y position)
        y_positions = [e['y'] for e in text_elements]
        baseline = np.median(y_positions)
        
        # Calculate threshold (you may need to tune this)
        heights = [e['height'] for e in text_elements]
        avg_height = np.mean(heights)
        threshold = avg_height * 0.3
        
        formatted_parts = []
        
        for element in text_elements:
            text = element['text']
            y = element['y']
            
            # Check if subscript or superscript
            if y > baseline + threshold:
                formatted_parts.append(f"_{text}")  # Subscript
            elif y < baseline - threshold:
                formatted_parts.append(f"^{text}")  # Superscript
            else:
                formatted_parts.append(text)
        
        return ' '.join(formatted_parts)
    
    def save_results(self, results, output_filename="ocr_results.json"):
        """
        Save OCR results to JSON file
        """
        output_path = self.output_dir / output_filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"Results saved to: {output_path}")
        return output_path
    
    def process_page(self, image_path):
        """
        Process a single page
        Returns: dictionary with all extracted data
        """
        print(f"\n{'='*60}")
        print(f"Processing: {image_path}")
        print(f"{'='*60}\n")
        
        # Step 1: Preprocess
        preprocessed = self.preprocess_image(image_path)
        
        # Step 2: Detect regions
        regions = self.detect_regions(preprocessed)
        
        # Step 3: Extract text with positions
        text_elements = self.extract_text_with_boxes(preprocessed)
        
        # Step 4: Detect formatting
        formatted_text = self.detect_formatting(text_elements)
        
        # Step 5: Basic text extraction (fallback)
        basic_text = self.extract_text_basic(preprocessed)
        
        results = {
            'image_path': str(image_path),
            'timestamp': datetime.now().isoformat(),
            'regions': regions,
            'text_elements': text_elements,
            'formatted_text': formatted_text,
            'basic_text': basic_text
        }
        
        return results


# Example usage
if __name__ == "__main__":
    print("Engineering Book OCR - Starter Template")
    print("=" * 60)
    
    # Initialize the OCR system
    ocr = EngineeringBookOCR(output_dir="ocr_output")
    
    # Example: Process a single image
    # Replace this with your actual image path
    image_path = "page004.jpeg"
    
    print(f"\nTo use this script:")
    print(f"1. Install requirements: pip install opencv-python pytesseract numpy")
    print(f"2. Install Tesseract OCR: https://github.com/tesseract-ocr/tesseract")
    print(f"3. Place your book page image as '{image_path}'")
    print(f"4. Run this script: python engineering_book_ocr.py")
    print(f"\nThe script will:")
    print(f"  - Preprocess the image for better OCR")
    print(f"  - Detect text regions, tables, and figures")
    print(f"  - Extract text with position data")
    print(f"  - Attempt to detect subscripts and superscripts")
    print(f"  - Save results to 'ocr_output' directory")
    
    # Uncomment below when you have an image ready
    results = ocr.process_page(image_path)
    ocr.save_results(results)
    # print("\n✓ Processing complete!")
    # print(f"\nFormatted text preview:")
    # print(results['formatted_text'][:500])  # First 500 characters