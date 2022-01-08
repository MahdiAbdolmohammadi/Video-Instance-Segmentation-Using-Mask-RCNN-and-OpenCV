import matplotlib.image as mpimg
import cv2
from utils import get_prediction, url_to_image, random_color_masks



def instance_segmentation(img_path, threshold=0.5, rect_th=3,
                          text_size=3, text_th=3, url=False):
    masks, boxes, pred_cls = get_prediction(img_path, threshold=threshold, url=url)
    if url:
        img = url_to_image(img_path) # If we have a url image
    else: # Local image
        img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # For working with RGB images instead of BGR
    for i in range(len(masks)):
        rgb_mask = random_color_masks(masks[i])
        img = cv2.addWeighted(img, 1, rgb_mask, 0.5, 0)
        cv2.rectangle(img, boxes[i][0], boxes[i][1], color=(0, 255, 0), thickness=rect_th)
        cv2.putText(img, pred_cls[i], boxes[i][0], cv2.FONT_HERSHEY_SIMPLEX, text_size, (0, 255, 0), thickness=text_th)
    return img, pred_cls, masks[i]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-input", "--input", help = "image input directori")
    parser.add_argument("-output", "--output", help = "image output directori")
    args = parser.parse_args()

    img, pred_classes, masks = instance_segmentation(args.input, rect_th=1,text_size=1, text_th=2)
    mpimg.imsave(args.output, img)