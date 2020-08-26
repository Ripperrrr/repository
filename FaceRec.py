from argparse import ArgumentParser
import numpy as np
from PIL import Image
import os
import sklearn.metrics.pairwise as pw
import cv2
import dlib


# define recognizer class
class Recognizer():
    # initial function, import two file path, cv, dlib, and threshold
    def __init__(self, src_path, target_path, threshold):
        self.src_path = src_path
        self.threshold = threshold
        self.filenames = []
        self.dst_rects_lst = []
        self.frame_face_num = {}

        # detect target image，extract face feature
        img = cv2.imread(target_path)
        self.target_images, self.target_rects = self.findFace(img.copy())
        self.target_features = self.feature_extraction(img.copy(), self.target_rects)

    # detect function
    def detect(self, img, cascade):
        # Call the face detection function of the cascade classifier and return the face frame
        rects = cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=3, flags=cv2.CASCADE_SCALE_IMAGE)
        if len(rects) == 0:
            return []
        rects[:, 2:] += rects[:, :2]
        return rects

    # draw rectangles
    def draw_rects(self, img, rects, color):
        for x1, y1, x2, y2 in rects:
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

    # Face detection main function
    def findFace(self, img, image_size=160):
        # Convert the image to a grayscale image and do histogram equalization to improve the image quality
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = cv2.equalizeHist(gray)

        # Create a cascade classifier using a face detector
        cascade_fn = os.path.join(CAS_PATH, haarcascade_frontalface_default)
        cascade = cv2.CascadeClassifier(cascade_fn)
        # detect face with cascade function
        rects = self.detect(gray, cascade)

        if len(rects) != 0:
            img_list = []
            # save all detected faces
            for rect in rects:
                vis = img[rect[1]:rect[3], rect[0]:rect[2], :]
                aligned = cv2.resize(vis, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
                img_list.append(aligned)

            # return images and rects
            images = np.stack(img_list)
            return images, rects
        else:
            return [], []

    # transfer rect as dlib rectangle object
    def convert_to_rect(self, rect):
        return dlib.rectangle(rect[0], rect[1], rect[2], rect[3])

    def feature_extraction(self, img, rects):
        feature = []

        if len(rects) > 0:
            # Define feature point predictor
            sp = dlib.shape_predictor(shape_predictor_68_face_landmarks)
            # Define dlib face recognition model
            facerec = dlib.face_recognition_model_v1(face_rec_model_path)
            for rect in rects:
                # Calculate the feature points of each face
                shape = sp(img, self.convert_to_rect(rect))

                # Import the original image and feature points into the face recognition model to obtain face features
                face_descriptor = facerec.compute_face_descriptor(img, shape)
                feature.append(face_descriptor)

            feature = np.array(feature)

        return feature

    # cosine similarity function
    def cosine_similarity(self, src_feature, target_feature):
        if len(src_feature) == 0 or len(target_feature) == 0:
            return np.empty(0)
        predicts = pw.cosine_similarity(src_feature, target_feature)
        return predicts

    # draw rectangles on image
    def draw_single_rect(self, img, rect, color):
        cv2.rectangle(img, (rect[0], rect[1]), (rect[2], rect[3]), color, 2)

    # Face recognition main function to identify target person from image
    def process(self, image):
        # First extract all the face features in the picture
        self.src_images, self.src_rects = self.findFace(image)
        self.src_features = self.feature_extraction(image, self.src_rects)

        # Calculate cosine similarity between features
        cosine_distances = self.cosine_similarity(self.src_features, self.target_features)

        # if not find, return
        if len(cosine_distances) == 0:
            return image

        # get subscript of max cosine distances
        index_x, index_y = np.where(cosine_distances == np.max(cosine_distances))

        # draw rectangles, Cosine distances and Euclidean distance
        for i in range(len(cosine_distances)):
            # check if cosine distance >= threshold
            if i == index_x and cosine_distances[index_x, index_y] >= self.threshold:
                # if find person, use green
                maxrate = cosine_distances[index_x, index_y]
                pen = (255, 0, 0)
            else:
                # if not target person, use red
                maxrate = cosine_distances[index_x, index_y]
                pen = (0, 0, 255)

            self.draw_single_rect(image, self.src_rects[i], pen)
            # cv2.putText(image, str(np.round(cosine_distances[i], 2)),
            #             (self.src_rects[i][0], self.src_rects[i][1] - 7),
            #             cv2.FONT_HERSHEY_DUPLEX, 2, pen)
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        return image, maxrate


def comparison_process(src, target, threshold=0.95):
    cls = Recognizer(src, target, threshold)
    if src.split('.')[-1] in ['jpg', 'JPG', 'jpeg', 'bmp', 'png']:
        src_img = cv2.imread(src)
        # process function
        out, rate = cls.process(src_img)
        return out, rate


CAS_PATH = r"./cascades"
shape_predictor_68_face_landmarks = "./shape_predictor_68_face_landmarks.dat"
face_rec_model_path = "./dlib_face_recognition_resnet_model_v1.dat"
haarcascade_frontalface_default = "haarcascade_frontalface_default.xml"

if __name__ == '__main__':
    out, rate = comparison_process("./chengshd/wyf2.jpeg", "./chengshd/wyf.jpg")
    print(rate)
