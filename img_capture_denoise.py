import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from skimage.util import random_noise
from sklearn.feature_extraction.image import extract_patches_2d, reconstruct_from_patches_2d
from sklearn.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio
from skimage.filters import unsharp_mask

def get_images():

    cam = cv2.VideoCapture(0)
    train_image = np.array([])
    test_image = np.array([])

    cv2.namedWindow("Camera")

    img_counter = 0

    while True:
        ret, frame = cam.read()
        if not ret:
            print("failed to grab frame")
            break
        cv2.imshow("Camera", frame)

        k = cv2.waitKey(1)
        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        elif k%256 == 32:
            # SPACE pressed
            if(img_counter == 0):
                train_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                test_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            img_name = "test_{}.jpeg".format(img_counter)
            cv2.imwrite(img_name, frame)
            print("{} written!".format(img_name))
            
            img_counter += 1

    cam.release()

    cv2.destroyAllWindows()

    return [train_image, test_image]

def train(image):
    # Get Heigh and Width
    height, width = image.shape

    font = cv2.FONT_HERSHEY_SIMPLEX 

    # Add random noise
    noisy_image = random_noise(image, mode='gaussian', clip=True, var=0.005)
    noisy_image = random_noise(noisy_image, mode='s&p', amount=0.001)

    # Extract patches from the original and noisy images
    noisy_patches = extract_patches_2d(noisy_image, patch_size=(7, 7), random_state=42)
    original_patches = extract_patches_2d(image, patch_size=(7, 7), random_state=42)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(noisy_patches.reshape(-1, 7*7), original_patches.reshape(-1, 7*7), test_size=0.2)

    # Initialize and train a Ridge Regression model
    model = Ridge(alpha=4)
    model.fit(X_train, y_train)

    # Calculate Error on test set
    print("Mean Squared Error on Test Set: " + str(mean_squared_error(y_test, model.predict(X_test))))

    # Denoise each patch using the model
    denoised_patches = model.predict(noisy_patches.reshape(-1, 7*7)).reshape(-1, 7, 7)

    # Reconstruct the image from the denoised patches
    denoised_image = reconstruct_from_patches_2d(denoised_patches, image_size=(height, width))
    
    # Sharpen the image
    sharpened_image = unsharp_mask(denoised_image, radius=2, amount=1)

    cv2.putText(noisy_image,  
                'PSNR: ' + str(peak_signal_noise_ratio(image, noisy_image)),  
                (50, 50),  
                font, 1,  
                (0, 255, 255),  
                2,  
                cv2.LINE_4) 
    cv2.imshow("Noisy Image for Training", noisy_image)
    cv2.imwrite("Noisy_Training_Image.jpeg", noisy_image*255.0)
    print('Noisy Image PSNR: ' + str(peak_signal_noise_ratio(image, noisy_image)))

    cv2.putText(denoised_image,  
                'PSNR: ' + str(peak_signal_noise_ratio(image, denoised_image)),  
                (50, 50),  
                font, 1,  
                (0, 255, 255),  
                2,  
                cv2.LINE_4) 
    cv2.imshow("Denoised Training Image", denoised_image)
    cv2.imwrite("Denoised_Training_Image.jpeg", denoised_image*255.0)
    print('Denoised Image PSNR: ' + str(peak_signal_noise_ratio(image, denoised_image)))

    cv2.putText(sharpened_image,  
                'PSNR: ' + str(peak_signal_noise_ratio(image, sharpened_image)),  
                (50, 50),  
                font, 1,  
                (0, 255, 255),  
                2,  
                cv2.LINE_4) 
    cv2.imshow("Sharpened Training Image", sharpened_image)
    cv2.imwrite("Sharpened_Training_Image.jpeg", sharpened_image*255.0)
    print("Sharpened Image PSNR: " + str(peak_signal_noise_ratio(image, sharpened_image)))

    return model

def test(image, model):
    height, width = image.shape
    
    # Extract Patches
    patches = extract_patches_2d(image, patch_size=(7, 7), random_state=42)

    # Denoise each patch using the model
    denoised_patches = model.predict(patches.reshape(-1, 7*7)).reshape(-1, 7, 7)

    # Reconstruct the image from the denoised patches
    denoised_image = reconstruct_from_patches_2d(denoised_patches, image_size=(height, width))

    cv2.imshow("Denoised Test Image", denoised_image)
    cv2.imwrite("Denoised_Test_Image.jpeg", denoised_image*255.0)

    # Sharpen the image
    sharpened_image = unsharp_mask(denoised_image, radius=2, amount=1)

    cv2.imshow("Sharpened Test Image", sharpened_image)
    cv2.imwrite("Sharpened_Test_Image.jpeg", sharpened_image*255.0)

train_image, test_image = get_images()

train_image = train_image / 255.0
test_image = test_image / 255.0

model = train(train_image)

test(test_image, model)

cv2.waitKey(0)


