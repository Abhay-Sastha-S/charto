import pytesseract
from PIL import Image
import sys
import os

def test_tesseract():
    """Test if Tesseract OCR engine is working properly"""
    try:
        # Set the tesseract executable path for Windows
        tesseract_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        if os.path.exists(tesseract_path):
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
            print(f"✅ Found Tesseract at: {tesseract_path}")
        else:
            print(f"❌ Tesseract not found at: {tesseract_path}")
            return False
        
        # Test 1: Check if pytesseract can find tesseract executable
        print("Testing Tesseract OCR engine...")
        print(f"Tesseract executable path: {pytesseract.pytesseract.tesseract_cmd}")
        
        # Test 2: Check tesseract version
        version = pytesseract.get_tesseract_version()
        print(f"Tesseract version: {version}")
        
        # Test 3: Create a simple test image with text
        print("\nCreating test image...")
        from PIL import Image, ImageDraw, ImageFont
        
        # Create a simple image with text
        img = Image.new('RGB', (200, 100), color='white')
        draw = ImageDraw.Draw(img)
        
        # Try to use a default font, fallback to basic if not available
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        draw.text((10, 30), "Hello World", fill='black', font=font)
        
        # Save test image
        test_image_path = "test_ocr_image.png"
        img.save(test_image_path)
        print(f"Test image saved as: {test_image_path}")
        
        # Test 4: Perform OCR on the test image
        print("\nPerforming OCR on test image...")
        text = pytesseract.image_to_string(img)
        print(f"OCR Result: '{text.strip()}'")
        
        # Test 5: Get detailed OCR data
        print("\nGetting detailed OCR data...")
        data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
        print(f"Number of detected text elements: {len([x for x in data['text'] if x.strip()])}")
        
        # Clean up test image
        if os.path.exists(test_image_path):
            os.remove(test_image_path)
            print(f"Cleaned up test image: {test_image_path}")
        
        print("\n✅ Tesseract OCR engine is working correctly!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error testing Tesseract: {str(e)}")
        print("\nTroubleshooting tips:")
        print("1. Make sure Tesseract is installed on your system")
        print("2. On Windows, you may need to add Tesseract to your PATH")
        print("3. Or set the tesseract_cmd path manually:")
        print("   pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'")
        return False

if __name__ == "__main__":
    test_tesseract()
