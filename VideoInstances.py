import cv2
from ImageInstances import instance_segmentation

def VidInstance(inputvideo, outputvideo, fps, width, height, frame_skip):
    # video = cv2.VideoWriter(outputvideo, cv2.VideoWriter_fourcc(*'DIVX'), fps, (width, height))
    video = cv2.VideoWriter("./examples/traficmask.avi", cv2.VideoWriter_fourcc(*'DIVX'), 5, (640, 360))
    cap = cv2.VideoCapture(inputvideo)

    i = 0
    frame_skip = int(frame_skip)
    frame_count = 0
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break
        if i > frame_skip - 1:
            frame_count += 1
            
            cv2.imwrite("framex.jpg", frame)
            filename = "framex.jpg"
            img, pred_classes, masks = instance_segmentation(filename,rect_th=1,text_size=1, text_th=2)

            # cv2.imwrite('Frame'+str(frame_count*frame_skip)+'.jpg', img)
            video.write(img)
            i = 0
            continue
        i += 1
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-inputvideo", "--inputvideo", help = "video input directori")
    parser.add_argument("-outputvideo", "--outputvideo", help = "video output directori")
    parser.add_argument("-fps", "--fps", help = "video output directori")
    parser.add_argument("-height", "--height", help = "video output height")
    parser.add_argument("-width", "--width", help = "video output width")
    parser.add_argument("-frame_skip", "--frame_skip", help = "drop frames of input video")
    args = parser.parse_args()

    VidInstance(args.inputvideo, args.outputvideo, args.fps, args.width, args.height, args.frame_skip)