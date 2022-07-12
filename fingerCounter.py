import cv2 as cv 
import mediapipe as mp 
import time

class handLandmarkDetector():
    def __init__(self,image_mode=False,max_hands=4, model_complexity = 1, min_detection_confidence=0.8,min_tracking_confidence=0.5):
        self.image_mode = image_mode
        self.max_hands=max_hands
        self.model_complexity = model_complexity
        self.min_detection_confidence=min_detection_confidence
        self.min_tracking_confidence=min_tracking_confidence

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.image_mode,self.max_hands,self.model_complexity,self.min_detection_confidence,self.min_tracking_confidence)
        self.mpDraw=mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        
    def detect_landmarks(self,image,draw=True,draw_connections=True,draw_default_style=False):
        imageRGB = cv.cvtColor(image,cv.COLOR_BGR2RGB)
        land_mark_data=[]
        hand_classified_landmarks=[[],[]]
        results = self.hands.process(imageRGB)
        landmarks=results.multi_hand_landmarks
        data=None
        if landmarks:
            for hand_landmarks in landmarks:
                for id,landmark in enumerate(hand_landmarks.landmark):
                    h,w,c = image.shape
                    px,py = int(landmark.x*w), int(landmark.y*h)
                    data=(id,px,py)
                    land_mark_data.append(data)
                if draw and not draw_connections:
                    self.mpDraw.draw_landmarks(image,hand_landmarks)
                elif draw and draw_connections and not draw_default_style:
                    self.mpDraw.draw_landmarks(image,hand_landmarks,self.mpHands.HAND_CONNECTIONS)
                elif draw and draw_connections and draw_default_style:
                    self.mpDraw.draw_landmarks(image,hand_landmarks,self.mpHands.HAND_CONNECTIONS,self.mp_drawing_styles.get_default_hand_landmarks_style(),self.mp_drawing_styles.get_default_hand_connections_style())
            if land_mark_data[0][1]>land_mark_data[4][1]:
                if len(land_mark_data)>20:
                    hand_classified_landmarks[1]=land_mark_data[0:21]
                    hand_classified_landmarks[0]=land_mark_data[21::]
                else:
                    hand_classified_landmarks[1]=land_mark_data[0:21]
            elif land_mark_data[4][1]>land_mark_data[0][1]:
                if len(land_mark_data)>20:
                    hand_classified_landmarks[0]=land_mark_data[0:21]
                    hand_classified_landmarks[1]=land_mark_data[21::]
                else:
                    hand_classified_landmarks[0]=land_mark_data[0:21]
        return hand_classified_landmarks,image

    def count_up_fingers(self,data):
        fingers=[[],[]]

        # data[1] = Right Hand
        # data[x-axis hand] [TIP/DIP/MCP Fingers] [y-axis hand]
        # Just focus on the middle point, the [TIP/DIP/MCP Fingers]
        # to define whether the fingers are opened or closed
        # because the [x-axis hand] and [y-axis hand] are the same for all fingers
        if len(data[1]) != 0:

            # point [3] is the thumb DIP (ruas ibu jari)
            # point [4] is the thumb TIP (ujung atas ibu jari)
            # IF the data at point [3] > the data at point [4]
            # then it's mean the thumb is opened and count as one finger
            # ELSE the thumb is closed and doesn't count
            if (data[1][3][1] > data[1][4][1]):
                fingers[1].append(1)
            else:
                fingers[1].append(0)
            
            # point [5] is the index finger MCP (ujung bawah telunjuk)
            # point [7] is the index finger DIP (ruas telunjuk)
            # point [8] is the index finger TIP (ujung atas telunjuk)
            # IF the data at point [5] > the data at point [8]
            # AND IF the data at point [7] > the data at point [8]
            # then it's mean the index finger is opened and count as one finger
            # ELSE the index finger is closed and doesn't count
            if (data[1][5][2] > data[1][8][2] and data[1][7][2] > data[1][8][2]) :
                fingers[1].append(1)
            else:
                fingers[1].append(0)
            
            # point [9] is the middle finger MCP (ujung bawah jari tengah)
            # point [11] is the middle finger DIP (ruas jari tengah)
            # point [12] is the middle finger TIP (ujung atas jari tengah)
            # IF the data at point [9] > the data at point [12]
            # AND IF the data at point [11] > the data at point [12]
            # then it's mean the middle finger is opened and count as one finger
            # ELSE the middle finger is closed and doesn't count
            if (data[1][9][2] > data[1][12][2] and data[1][11][2] > data[1][12][2]) :
                fingers[1].append(1)
            else:
                fingers[1].append(0)
            
            # point [13] is the ring finger MCP (ujung bawah jari manis)
            # point [15] is the ring finger DIP (ruas jari manis)
            # point [16] is the ring finger TIP (ujung atas jari manis)
            # IF the data at point [13] > the data at point [16]
            # AND IF the data at point [15] > the data at point [16]
            # then it's mean the ring finger is opened and count as one finger
            # ELSE the ring finger is closed and doesn't count
            if (data[1][13][2] > data[1][16][2] and data[1][15][2] > data[1][16][2]) :
                fingers[1].append(1)
            else:
                fingers[1].append(0)
            
            # point [17] is the pingky finger MCP (ujung bawah jari kelingking)
            # point [19] is the pingky finger DIP (ruas jari kelingking)
            # point [20] is the pingky finger TIP (ujung atas jari kelingking)
            # IF the data at point [17] > the data at point [20]
            # AND IF the data at point [19] > the data at point [20]
            # then it's mean the pingky finger is opened and count as one finger
            # ELSE the pingky finger is closed and doesn't count
            if (data[1][17][2] > data[1][20][2] and data[1][19][2] > data[1][20][2]) :
                fingers[1].append(1)
            else:
                fingers[1].append(0)



        # data[0] = Left Hand
        # data[x-axis hand] [TIP/DIP/MCP Fingers] [y-axis hand]
        # Just focus on the middle point, the [TIP/DIP/MCP Fingers]
        # to define whether the fingers are opened or closed
        # because the [x-axis hand] and [y-axis hand] are the same for all fingers
        if len(data[0]) != 0:

            # point [3] is the thumb DIP (ruas ibu jari)
            # point [4] is the thumb TIP (ujung atas ibu jari)
            # IF the data at point [3] < the data at point [4]
            # then it's mean the thumb is opened and count as one finger
            # ELSE the thumb is closed and doesn't count
            if (data[0][3][1] < data[0][4][1]):
                fingers[0].append(1)
            else:
                fingers[0].append(0)

            # point [5] is the index finger MCP (ujung bawah telunjuk)
            # point [7] is the index finger DIP (ruas telunjuk)
            # point [8] is the index finger TIP (ujung atas telunjuk)
            # IF the data at point [5] > the data at point [8]
            # AND IF the data at point [7] > the data at point [8]
            # then it's mean the index finger is opened and count as one finger
            # ELSE the index finger is closed and doesn't count
            if (data[0][5][2] > data[0][8][2] and data[0][7][2] > data[0][8][2]) :
                fingers[0].append(1)
            else:
                fingers[0].append(0)
            
            # point [9] is the middle finger MCP (ujung bawah jari tengah)
            # point [11] is the middle finger DIP (ruas jari tengah)
            # point [12] is the middle finger TIP (ujung atas jari tengah)
            # IF the data at point [9] > the data at point [12]
            # AND IF the data at point [11] > the data at point [12]
            # then it's mean the middle finger is opened and count as one finger
            # ELSE the middle finger is closed and doesn't count
            if (data[0][9][2] > data[0][12][2] and data[0][11][2] > data[0][12][2]) :
                fingers[0].append(1)
            else:
                fingers[0].append(0)
            
            # point [13] is the ring finger MCP (ujung bawah jari manis)
            # point [15] is the ring finger DIP (ruas jari manis)
            # point [16] is the ring finger TIP (ujung atas jari manis)
            # IF the data at point [13] > the data at point [16]
            # AND IF the data at point [15] > the data at point [16]
            # then it's mean the ring finger is opened and count as one finger
            # ELSE the ring finger is closed and doesn't count
            if (data[0][13][2] > data[0][16][2] and data[0][15][2] > data[0][16][2]) :
                fingers[0].append(1)
            else:
                fingers[0].append(0)
            
            # point [17] is the pingky finger MCP (ujung bawah jari kelingking)
            # point [19] is the pingky finger DIP (ruas jari kelingking)
            # point [20] is the pingky finger TIP (ujung atas jari kelingking)
            # IF the data at point [17] > the data at point [20]
            # AND IF the data at point [19] > the data at point [20]
            # then it's mean the pingky finger is opened and count as one finger
            # ELSE the pingky finger is closed and doesn't count
            if (data[0][17][2] > data[0][20][2] and data[0][19][2] > data[0][20][2]) :
                fingers[0].append(1)
            else:
                fingers[0].append(0)
            
        return fingers

def main():
    
    capture = cv.VideoCapture(0)

    # Using web cam
    
    current_time = 0
    previous_time = 0

    hand_detector = handLandmarkDetector()

    while True:
        ret,image = capture.read()
        if image is None :
            continue
        image = cv.flip(image,1)
        image = cv.resize(image,(750, 550))

        landmarks,image = hand_detector.detect_landmarks(image,draw_default_style=False)

        fingers=hand_detector.count_up_fingers(landmarks)
        fingers_up = int(fingers[0].count(1)) + int(fingers[1].count(1))
        print(fingers_up)

        current_time = time.time()
        fps = 1/(current_time-previous_time)
        previous_time = current_time
        fps2text = 'FPS : '+str(int(fps))

        cv.rectangle(image,(10,10),(250,170),(0,0,0),-1)
        cv.putText(image,str(fingers_up),(10,147),cv.FONT_HERSHEY_SIMPLEX,5.5,(255,255,255),10)
        cv.putText(image,fps2text,(20,230),cv.FONT_HERSHEY_COMPLEX,1.5,(0,0,255),3)
        cv.imshow("Finger Counter Camera",image)
        if cv.waitKey(1)==27:
            break
    cv.destroyAllWindows()


if __name__ == '__main__':
     main()