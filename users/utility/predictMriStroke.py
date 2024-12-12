def start_process(imagepath):
    import os
    import cv2
    import imutils
    from keras.models import load_model
    from django.conf import settings
    img_path = os.path.join(settings.MEDIA_ROOT,imagepath)
    model_path = os.path.join(settings.MEDIA_ROOT,'brain_tumor_detector.h5')
    model = load_model(model_path,compile=False)
    image = cv2.imread(img_path)
    print('*'*50)
    print(model)
    
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Threshold the image, then perform a series of erosions +
    # dilations to remove any small regions of noise
    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

    # Find contours in thresholded image, then grab the largest one
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)

    # Find the extreme points
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])

    # crop new image out of the original image using the four extreme points (left, right, top, bottom)
    new_image = image[extTop[1]:extBot[1], extLeft[0]:extRight[0]]

    image = cv2.resize(new_image, dsize=(240, 240), interpolation=cv2.INTER_CUBIC)
    image = image / 255.

    image = image.reshape((1, 240, 240, 3))

    res = model.predict(image)

    return res