import cv2 as cv
import numpy as np

def equalize_image(image):
    # Convert the image to grayscale
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
    # Equalize the histogram of the grayscale image
    equalized_image = cv.equalizeHist(gray_image)
    
    # Convert the grayscale equalized image to color (3 channels)
    equalized_image_color = cv.cvtColor(equalized_image, cv.COLOR_GRAY2BGR)
    
    return equalized_image_color

# Initialize the webcam
cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # Perform histogram equalization
    equalized_frame = equalize_image(frame)
    
    # Display the original and equalized images side by side
    stacked_images = np.hstack((frame, equalized_frame))
    
    cv.imshow("Original vs Equalized", stacked_images)
    
    if cv.waitKey(1) & 0xFF == ord('q'):
        cv.imwrite("Original vs Equalized.jpg", stacked_images)
        break

# Release the webcam and close the windows
cap.release()
cv.destroyAllWindows()
