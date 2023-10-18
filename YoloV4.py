import cv2
import numpy as np
# load yolov4 with a opencv

class YoloV4:
    # initialize yolov4
    def __init__(self):
        self.net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")
        self.classes = []
        with open("coco.names", "r") as f:
            self.classes = [line.strip() for line in f.readlines()]
        self.layer_names = self.net.getLayerNames()       
        self.output_layers = self.net.getUnconnectedOutLayersNames()
        
        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))
    # detect objects
    def detect(self, img):
        height, width, channels = img.shape
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        font = cv2.FONT_HERSHEY_PLAIN
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(self.classes[class_ids[i]])
                color = self.colors[class_ids[i] % len(self.colors)]
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img, label, (x, y + 30), font, 3, color, 3)
        return img

# using above class make a test example
if __name__ == "__main__":
    # load image    
    img = cv2.imread("./cat.jpg")
    
    # detect objects
    yolo = YoloV4()
    result_img = yolo.detect(img)
    
    # show image
    cv2.imshow("Image", result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()