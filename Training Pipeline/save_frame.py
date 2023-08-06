import cv2

path = r'D:\AutoVisionPR Data\Testing\Videos'
video = r'\DN 7 Pitesti - Valcea.mp4'
cap = cv2.VideoCapture(path + video)

counter = 0
print('Enter the time of the frame you want to save (minute:second)')

while True:
    text = input()
    if text == 'q':
        break

    frame_time = text.split(':')
    minute = int(frame_time[0])
    second = int(frame_time[1])

    fps = cap.get(cv2.CAP_PROP_FPS)  # get the frames per second
    frame_no = int(fps * 60 * minute + fps * second)  # calculate the frame number

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    ret, frame = cap.read()

    if ret:
        cv2.imwrite(f'frame{counter}.png', frame)
        print(f'Frame from {text} saved as image.')

cap.release()
cv2.destroyAllWindows()
