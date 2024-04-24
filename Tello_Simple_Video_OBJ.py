import random
import cv2
import numpy as np
from ultralytics import YOLO
import threading
import pygame
from djitellopy import Tello
import time # For video and controller fps control

# Initialize Pygame and Joystick
pygame.init()
pygame.joystick.init()

# Check for joystick count
joystick_count = pygame.joystick.get_count()
if joystick_count == 0:
    print("No joystick detected")
    exit()

# Initialize the Joystick
joystick = pygame.joystick.Joystick(0)

joystick.init()
##-----------------------------------------------------------##
###                     Connect to Tello                    ###
##-----------------------------------------------------------##
tello = Tello()

tello.connect()

speed = 100 # 10 - 100 cm/s

tello.set_speed(speed)



def joystick_control_thread():
    try:
        while True:
            pygame.event.pump()

            ##--------------------------------------------------##
            ###                 Joysticks                      ###
            ##--------------------------------------------------##
            # Joystick axes Normalized to -100 to 100
            left_joystick_x = int(joystick.get_axis(0)*500)
            left_joystick_y = int(joystick.get_axis(1)*500)
            right_joystick_x = int(joystick.get_axis(2)*500)
            right_joystick_y = int(joystick.get_axis(3)*500)

            left_right_velocity = right_joystick_x
            forward_backward_velocity = -right_joystick_y
            yaw_velocity = left_joystick_x
            up_down_velocity = -left_joystick_y

            tello.send_rc_control(left_right_velocity, 
                            forward_backward_velocity, up_down_velocity, yaw_velocity)

            ##--------------------------------------------------##
            ###                  Buttons                       ###
            ##--------------------------------------------------##
            for event in pygame.event.get():
                if event.type == pygame.JOYBUTTONDOWN: 
                    if event.button == 6:
                        print("Three lines button pressed")
                        print("Connecting")
                        tello.connect()

                    elif event.button == 0:
                        print("X button pressed")
                        print("Landing")
                        tello.land()

                    elif event.button == 3:
                        print("Triangle button pressed")
                        print("Take off initiated")
                        tello.takeoff()
    except KeyboardInterrupt:
        print("Exiting...")
        tello.land()
    finally:
        pygame.quit()


# Start the joystick control thread
joystick_thread = threading.Thread(target=joystick_control_thread)
joystick_thread.start()

##--------------------------------------------------##
###             Object Detection + Video            ###
##--------------------------------------------------##
# opening the file in read mode
# classes in a txt file
my_file = open("utils/coco.txt", "r") 
# reading the file
data = my_file.read()
# replacing end splitting the text | when newline ('\n') is seen.
# create a list of all classes
class_list = data.split("\n")
my_file.close()

# Generate random colors for class list
# used to draW boxes around detections
detection_colors = []
for i in range(len(class_list)):
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    detection_colors.append((b, g, r))

# load a pretrained YOLOv8n model
model = YOLO("weights/yolov8n.pt", "v8")

tello.streamon()
# Grab a frame from the video stream

while True:
    frame = tello.get_frame_read().frame
    frame_wid = 640
    frame_hyt = 480
    frame = cv2.resize(frame, (frame_wid, frame_hyt))
    #cv2.imshow('Tello Stream', frame)
    battery_level = tello.get_battery()
    battxt = f"Battery: {battery_level}%"
    cv2.putText(frame, battxt, (5, 480 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    # Predict on image
    detect_params = model.predict(source=[frame], conf=0.45, save=False)
    DP = detect_params[0].numpy()
    if len(DP) != 0:
        for i in range(len(detect_params[0])):
            print(i)

        boxes = detect_params[0].boxes
        box = boxes[i]  # returns one box
        clsID = box.cls.numpy()[0]
        conf = box.conf.numpy()[0]
        bb = box.xyxy.numpy()[0]
        # draw bounding box around detection
        cv2.rectangle(
            frame,
            (int(bb[0]), int(bb[1])),
            (int(bb[2]), int(bb[3])),
            detection_colors[int(clsID)],
            3,
        )

        # Display class name and confidence and battery?
        font = cv2.FONT_HERSHEY_COMPLEX
        cv2.putText(
            frame,
            class_list[int(clsID)] + " " + str(round(conf, 3)) + "%",
            (int(bb[0]), int(bb[1]) - 10),
            font,
            1,
            (255, 255, 255),
            2,
        )
    cv2.imshow("ObjectDetection", frame)


    # If 'q' is pressed, exit the while loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    
# Clean up
cv2.destroyAllWindows()
tello.streamoff()
