# 1: Import our plugins 
import cv2 
import time
import requests
from picamera2 import Picamera2

# 2: Set our variables 
DISCORD_WEBHOOK = ""
COOLDOWN_SECONDS = 5
last_sent_time = 0 
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Init our PiCamera 
picam = Picamera2()
config = picam.create_preview_configuration({'size':(320, 240), 'format':'RGB888'})
picam.configure(config)
picam.start()

print("Starting Face Detection...")

try:
    while True:
        #1 : Get current frame 
        frame = picam.capture_array()
        #2: Convert it to grayscale 
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        #3: Detect faces in grayscale format
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30)
        )

        #4: If faces are detected, send them off to discord
        if len(faces) > 0:
            timestamp = time.strftime("%H:%M:%S")
            elapsed_time = time.time() - last_sent_time
            if elapsed_time > COOLDOWN_SECONDS:
                last_sent_time = time.time()

                # Image saving format 
                image_format = f"detected_face_{timestamp}.jpg"
                cv2.imwrite(image_format, frame)

                # Send the image to discord 
                with open(image_format, "rb") as f:
                    discord_file = {'files':(image_format, f, 'image/jpeg')}
                    discord_payload = {'content':f"**FACE DETECTED**\nTimestamp: {timestamp}"}

                    response = requests.post(DISCORD_WEBHOOK, data=discord_payload, files=discord_file)
                    if response.status_code == 200:
                        print(f"File sent successfully to Discord {image_format}")
                    else:
                        print(f"Failed to send the file to discord :(")

        time.sleep(0.1)
except KeyboardInterrupt:
    print("\nCntrl+C Detected, exiting loop...")

finally:
    picam.stop()
    print("Exiting program...")
