import multiprocessing
import os
import urllib.request as urlreq
import networkx as nx
from networkx.readwrite import json_graph
import cv2
import numpy as np
from tqdm import tqdm
from face_recognition import face_locations, face_landmarks
from skimage.util import img_as_float
import mediapipe as mp
import config as config
from skimage.segmentation import slic
from skimage.future import graph
from skimage import color, segmentation
from utils.funcs import _weight_mean_color, merge_mean_color, weight_boundary, merge_boundary
# from test import plot_graph_from_image,get_graph_from_image
from sklearn import preprocessing
import h5py


def Deepphys_preprocess_Video(path, flag):
    '''
    :param path: dataset path
    :param flag: face detect flag
    :return: [:,:,:0-2] : motion diff frame
             [:,:,:,3-5] : normalized frame
    '''
    cap = cv2.VideoCapture(path)
    frame_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    raw_video = np.empty((frame_total - 1, 36, 36, 6))
    prev_frame = None
    j = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if frame is None:
            break
        if flag:
            rst, crop_frame = faceDetection(frame)
            if not rst:  # can't detect face
                return False, None
        else:
            crop_frame = frame[:, int(width / 2) - int(height / 2 + 1):int(height / 2) + int(width / 2), :]

        crop_frame = cv2.resize(crop_frame, dsize=(36, 36), interpolation=cv2.INTER_AREA)
        crop_frame = generate_Floatimage(crop_frame)

        if prev_frame is None:
            prev_frame = crop_frame
            continue
        raw_video[j, :, :, :3], raw_video[j, :, :, -3:] = preprocess_Image(prev_frame, crop_frame)
        prev_frame = crop_frame
        j += 1
    raw_video[:, :, :, 3] = video_normalize(raw_video[:, :, :, 3])
    cap.release()

    return True, raw_video


def PhysNet_preprocess_Video(path, flag):
    '''
    :param path: dataset path
    :param flag: face detect flag
    :return:
    '''
    set = 128
    div = 32
    if path.__contains__("png"):
        path = path[:-3]
        data = os.listdir(path)[:-1]
        data.sort()
        frame_total = len(data)
        raw_video = np.empty((frame_total, set, set, 3))
        j = 0

        if flag == 2:
            detector = FaceMeshDetector(maxFaces=2)
        elif flag == 3:
            detector = get_haarcascade()

        mp_face_detection = mp.solutions.face_detection
        with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
            # ret, frame = cap.read()
            for img in data:
                frame = cv2.imread(path + "/" + img)
                height,width,channel = frame.shape
                frame.flags.writeable = False
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_detection.process(frame)
                if len(results.detections) > 0:
                    bbox = results.detections[0].location_data.relative_bounding_box
                    crop_frame = frame[int(height * bbox.ymin):int(height * (bbox.ymin + bbox.height)),
                                 int(width * bbox.xmin):int(width * (bbox.xmin + bbox.width))]
                else:
                    mask = np.zeros(frame.shape[:2], dtype="uint8")
                    crop_frame = cv2.bitwise_and(frame, frame, mask=mask)
                crop_frame = cv2.resize(crop_frame, dsize=(set, set), interpolation=cv2.INTER_CUBIC)
                crop_frame = generate_Floatimage(crop_frame)

                raw_video[j] = crop_frame
                j += 1
            # pbar.update(1)
        print("END")
        # cap.release()

        # for img in data:
        #     frame = cv2.imread(path + "/" + img)
        #     if flag == 1:
        #         rst, crop_frame = faceDetection(frame)
        #         if len(faces) is not 0:
        #             (x, y, w, d) = faces[0]
        #             frame_cropped = frame[y:(y + d), x:(x + w)]
        #             # eyes = eye_detector.detectMultiScale(frame_cropped, 1.2, 3)
        #             crop_frame = frame_cropped
        #         else:
        #             # The problemis that this doesn't get cropped :/
        #             # (x, y, w, d) = (308, 189, 215, 215)
        #             # frame_masked = frame[y:(y + d), x:(x + w)]
        #
        #             # print("face detection failed, image frame will be masked")
        #             mask = np.zeros(frame.shape[:2], dtype="uint8")
        #             crop_frame = cv2.bitwise_and(frame, frame, mask=mask)
        #
        #         if not rst:  # can't detect face
        #             return False, None
        #     elif flag == 2:
        #         f, dot = crop_mediapipe(detector, frame)
        #         # bin_mask =  '1001_0000_0001_0000_0000_0000_0000_1100'
        #         bin_mask = '0011000000000000000100000001001'
        #         view, remove = make_mask(dot)
        #         crop_frame = generate_maks(f, view, remove)
        #
        #         _, dot = detector.findFaceMesh(f)
        #     elif flag == 3:
        #         faces = detector.detectMultiScale(frame, 1.3, 5)
        #         # faces = detector.faceDetection
        #         if len(faces) is not 0:
        #             (x, y, w, d) = faces[0]
        #             frame_cropped = frame[y:(y + d), x:(x + w)]
        #             # eyes = eye_detector.detectMultiScale(frame_cropped, 1.2, 3)
        #             crop_frame = frame_cropped
        #         else:
        #             # The problemis that this doesn't get cropped :/
        #             # (x, y, w, d) = (308, 189, 215, 215)
        #             # frame_masked = frame[y:(y + d), x:(x + w)]
        #
        #             # print("face detection failed, image frame will be masked")
        #             mask = np.zeros(frame.shape[:2], dtype="uint8")
        #             crop_frame = cv2.bitwise_and(frame, frame, mask=mask)
        #         # crop_frame.append(frame_resized)
        #
        #     crop_frame = cv2.resize(crop_frame, dsize=(set, set), interpolation=cv2.INTER_CUBIC)
        #     crop_frame = generate_Floatimage(crop_frame)
        #
        #     raw_video[j] = crop_frame
        #     j += 1

        print("A")
    else:
        cap = cv2.VideoCapture(path)
        frame_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        raw_video = np.empty((frame_total, set, set, 3))

        j = 0

        detector = None

        mp_face_detection = mp.solutions.face_detection
        with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
            ret, frame = cap.read()
            frame.flags.writeable = False
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(frame)
            if len(results.detections) >0:
                bbox = results.detections[0].location_data.relative_bounding_box
                crop_frame = frame[int(height*bbox.ymin):int(height*(bbox.ymin+bbox.height)),int(width*bbox.xmin):int(width*(bbox.xmin + bbox.width))]
            else:
                mask = np.zeros(frame.shape[:2], dtype="uint8")
                crop_frame = cv2.bitwise_and(frame, frame, mask=mask)
            crop_frame = cv2.resize(crop_frame, dsize=(set, set), interpolation=cv2.INTER_CUBIC)
            crop_frame = generate_Floatimage(crop_frame)

            raw_video[j] = crop_frame
            j += 1
            # pbar.update(1)
        cap.release()
            # print("A")

        # if flag == 2:
        #     detector = FaceMeshDetector(maxFaces=2)
        # elif flag == 3:
        #     detector = get_haarcascade()

        # with tqdm(total=frame_total, position=0, leave=True, desc=path) as pbar:
        #     while cap.isOpened():
        #         ret, frame = cap.read()
        #         if frame is None:
        #             break
        #         if flag == 1:
        #             rst, crop_frame = faceDetection(frame)
        #             if not rst:  # can't detect face
        #                 # The problemis that this doesn't get cropped :/
        #                 # (x, y, w, d) = (308, 189, 215, 215)
        #                 # frame_masked = frame[y:(y + d), x:(x + w)]
        #
        #                 # print("face detection failed, image frame will be masked")
        #                 mask = np.zeros(frame.shape[:2], dtype="uint8")
        #                 crop_frame = cv2.bitwise_and(frame, frame, mask=mask)
        #         elif flag == 2:
        #             crop_frame, _ = crop_mediapipe(detector, frame)
        #             if crop_frame == frame:
        #                 mask = np.zeros(frame.shape[:2], dtype="uint8")
        #                 crop_frame = cv2.bitwise_and(frame, frame, mask=mask)
        #             # bin_mask =  '1001_0000_0001_0000_0000_0000_0000_1100'
        #             bin_mask = '0011000000000000000100000001001'
        #             # view,remove = make_mask(dot)
        #             # crop_frame = generate_maks(f,view,remove)
        #
        #             # _, dot = detector.findFaceMesh(f)
        #         elif flag == 3:
        #             faces = detector.detectMultiScale(frame, 1.3, 5)
        #             if len(faces) is not 0:
        #                 (x, y, w, d) = faces[0]
        #                 frame_cropped = frame[y:(y + d), x:(x + w)]
        #                 # eyes = eye_detector.detectMultiScale(frame_cropped, 1.2, 3)
        #                 crop_frame = frame_cropped
        #             else:
        #                 # The problemis that this doesn't get cropped :/
        #                 # (x, y, w, d) = (308, 189, 215, 215)
        #                 # frame_masked = frame[y:(y + d), x:(x + w)]
        #
        #                 # print("face detection failed, image frame will be masked")
        #                 mask = np.zeros(frame.shape[:2], dtype="uint8")
        #                 crop_frame = cv2.bitwise_and(frame, frame, mask=mask)
        #             # crop_frame.append(frame_resized)
        #
        #         else:
        #             crop_frame = frame[:, int(width / 2) - int(height / 2 + 1):int(height / 2) + int(width / 2), :]
        #
        #         crop_frame = cv2.resize(crop_frame, dsize=(set, set), interpolation=cv2.INTER_CUBIC)
        #         crop_frame = generate_Floatimage(crop_frame)
        #
        #         raw_video[j] = crop_frame
        #         j += 1
        #         pbar.update(1)
        #     cap.release()

    split_raw_video = np.zeros(((frame_total // 32), 32, 128, 128, 3))
    index = 0
    for x in range(frame_total // 32):
        split_raw_video[x] = raw_video[index:index + 32]
        index = index + div

    return True, split_raw_video, frame_total


def RTNet_preprocess_Video(path, flag):
    '''
    :param path: dataset path
    :param flag: face detect flag
    :return:
    '''
    cap = cv2.VideoCapture(path)
    frame_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    preprocessed_video = np.empty((frame_total, 36, 36, 6))
    j = 0
    with tqdm(total=frame_total, position=0, leave=True, desc=path) as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if frame is None:
                break
            if flag:  # TODO: make flag == false option
                rst, crop_frame, mask = faceLandmarks(frame)
                if not rst:  # can't detect face
                    return False, None
            else:
                crop_frame = frame[:, int(width / 2) - int(height / 2 + 1):int(height / 2) + int(width / 2), :]

            crop_frame = cv2.resize(crop_frame, dsize=(128, 128), interpolation=cv2.INTER_AREA)
            crop_frame = generate_Floatimage(crop_frame)

            mask = cv2.resize(mask, dsize=(128, 128), interpolation=cv2.INTER_AREA)
            mask = generate_Floatimage(mask)

            preprocessed_video[j:, :, :, 3], preprocessed_video[j:, :, :, -3] = crop_frame, mask

            j += 1
            pbar.update(1)
        cap.release()

    preprocessed_video[:, :, :, 3] = video_normalize(preprocessed_video[:, :, :, 3])

    return True, preprocessed_video


def GCN_preprocess_Video(path, flag):
    '''
       :param path: dataset path
       :param flag: face detect flag
       :return:
       '''
    maps, sliding_window_stride = preprocess_video_to_st_maps(path, output_shape=(180, 180))
    return True, maps, sliding_window_stride


def Axis_preprocess_Video(path, flag):
    '''
       :param path: dataset path
       :param flag: face detect flag
       :return:
       '''
    preprocess_video_to_st_maps(path, (256, 256))
    maps, sliding_window_stride, num_frames, stacked_ptts = preprocess_video_to_st_maps(path, output_shape=(180, 180))
    # bvp,sliding,frames,ptt
    return True, maps, sliding_window_stride, num_frames, stacked_ptts


def faceLandmarks(frame):
    '''
    :param frame: one frame
    :return: landmarks
    '''
    resized_frame = cv2.resize(frame, (0, 0), fx=1, fy=1)
    grayscale_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
    face_location = face_locations(resized_frame)
    if len(face_location) == 0:  # can't detect face
        return False, None, None
    face_landmark_list = face_landmarks(resized_frame)
    i = 0
    center_list = []
    for face_landmark in face_landmark_list:
        for facial_feature in face_landmark.keys():
            for center in face_landmark[facial_feature]:
                center_list.append(center)
                i = i + 1
    pt = np.array([center_list[2], center_list[3], center_list[31]])
    pt1 = np.array([center_list[13], center_list[14], center_list[35]])
    pt2 = np.array([center_list[6], center_list[7], center_list[65]])
    pt3 = np.array([center_list[9], center_list[10], center_list[61]])
    dst = cv2.fillConvexPoly(grayscale_frame, pt, color=(255, 255, 255))
    dst = cv2.fillConvexPoly(dst, pt1, color=(255, 255, 255))
    dst = cv2.fillConvexPoly(dst, pt2, color=(255, 255, 255))
    dst = cv2.fillConvexPoly(dst, pt3, color=(255, 255, 255))
    for i in range(dst.shape[0]):
        for j in range(dst.shape[1]):
            if dst[i][j] != 255:
                dst[i][j] = 0
    top, right, bottom, left = face_location[0]
    dst = resized_frame[top:bottom, left:right]
    mask = grayscale_frame[top:bottom, left:right]
    # test = cv2.bitwise_and(dst,dst,mask=mask)

    return True, dst, mask


def faceDetection(frame):
    '''
    :param frame: one frame
    :return: cropped face image
    '''
    resized_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    face_location = face_locations(resized_frame)
    if len(face_location) == 0:  # can't detect face
        return False, None
    top, right, bottom, left = face_location[0]
    dst = resized_frame[top:bottom, left:right]
    return True, dst


def generate_Floatimage(frame):
    '''
    :param frame: roi frame
    :return: float value frame [0 ~ 1.0]
    '''
    dst = img_as_float(frame)
    # 왜 있지??
    dst = cv2.cvtColor(dst.astype('float32'), cv2.COLOR_BGR2RGB)
    dst[dst > 1] = 1
    dst[dst < 0] = 0
    return dst


def generate_MotionDifference(prev_frame, crop_frame):
    '''
    :param prev_frame: previous frame
    :param crop_frame: current frame
    :return: motion diff frame
    '''
    # motion input
    motion_input = (crop_frame - prev_frame) / (crop_frame + prev_frame)
    # TODO : need to diminish outliers [ clipping ]
    # motion_input = motion_input / np.std(motion_input)
    # TODO : do not divide each D frame, modify divide whole video's unit standard deviation
    return motion_input


def normalize_Image(frame):
    '''
    :param frame: image
    :return: normalized_frame
    '''
    return frame / np.std(frame)


def preprocess_Image(prev_frame, crop_frame):
    '''
    :param prev_frame: previous frame
    :param crop_frame: current frame
    :return: motion_differnceframe, normalized_frame
    '''
    return generate_MotionDifference(prev_frame, crop_frame), normalize_Image(prev_frame)


def ci99(motion_diff):
    max99 = np.mean(motion_diff) + (2.58 * (np.std(motion_diff) / np.sqrt(len(motion_diff))))
    min99 = np.mean(motion_diff) - (2.58 * (np.std(motion_diff) / np.sqrt(len(motion_diff))))
    motion_diff[motion_diff > max99] = max99
    motion_diff[motion_diff < min99] = min99
    return motion_diff


def video_normalize(channel):
    channel /= np.std(channel)
    return channel


class FaceMeshDetector:

    def __init__(self, staticMode=False, maxFaces=2, minDetectionCon=0.5, minTrackCon=0.5):

        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.minDetectionCon = minDetectionCon
        self.minTrackCon = minTrackCon

        self.mpDraw = mp.solutions.drawing_utils

        self.mpFaceDetection = mp.solutions.face_detection
        self.faceDetection = self.mpFaceDetection.FaceDetection()

        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.staticMode, self.maxFaces,
                                                 self.minDetectionCon, self.minTrackCon)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=2)

    def findFaceMesh(self, img, draw=True):
        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(img)
        # self.faces = self.faceDetection.process(img)

        faces = []
        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:

                face = []
                for id, lm in enumerate(faceLms.landmark):
                    # print(lm)
                    ih, iw, ic = img.shape
                    x, y = int(lm.x * iw), int(lm.y * ih)
                    # cv2.putText(img, str(id), (x, y), cv2.FONT_HERSHEY_PLAIN,
                    #           0.7, (0, 255, 0), 1)

                    # print(id,x,y)
                    face.append([x, y])
                faces.append(face)
        return img, faces


def avg(a, b):
    return [(int)((x + y) / 2) for x, y in zip(a, b)]


def crop_mediapipe(detector, frame):
    _, dot = detector.findFaceMesh(frame)
    if len(dot) > 0:
        x_min = min(np.array(dot[0][:]).T[0])
        y_min = min(np.array(dot[0][:]).T[1])
        x_max = max(np.array(dot[0][:]).T[0])
        y_max = max(np.array(dot[0][:]).T[1])
        x_center = (int)((x_min + x_max) / 2)
        y_center = (int)((y_min + y_max) / 2)
        if (x_max - x_min) > (y_max - y_min):
            w_2 = (int)((x_max - x_min) / 2)
        else:
            w_2 = (int)((y_max - y_min) / 2)
        f = frame[y_center - w_2 - 10:y_center + w_2 + 10, x_center - w_2 - 10:x_center + w_2 + 10]
        _, dot = detector.findFaceMesh(f)
        return f, dot[0]
    else:
        return frame, _


def make_specific_mask(bin_mask, dot):
    '''
    :param num:
    0 :  lower_cheek_left,          1 :  lower_cheek_right,          2 :  Malar_left
    3 :  Malar_right,               4 :  Marionette_Fold_left        5 :  Marionette_Fold_right
    6 :  chine                      7 :  Nasolabial_Fold_left        8 :  Nasolabial_Fold_right
    9 :  Lower_NasaL_Sidewall       10:  Upper_Lip_left              11:  Upper_Lip_right
    12:  Philtrum                   13:  Nasal_Tip                   14:  Lower_Nasal_Dorsum
    15: Lower_Nasal_Sidewall_left   16:  Lower_Nasal_Sidewall_right  17:  Mid_Nasal_Sidewall_left
    18: Mid_Nasal_Sidewall_right    19:  Upper_Nasal_Dorsum          20:  Glabella
    21: Lower_Lateral_Forehead_left 22: Lower_Lateral_Forehead_right 23: temporal_lobe_left
    24: temporal_lobe_right         25: eye_left                     26: eye_right
    27: Lower_Medial_Forehead       28: Upper_Lateral_Forehead_left  29: Upper_Lateral_Forehead_right
    30: Upper_Medial_Forehead
    :param dot:
    :return:
    '''
    mask_list = []
    for (idx, bin) in enumerate(bin_mask):
        if bin == '1':
            if idx == 0:  # lower_cheek_left
                mask_list.append(np.array(
                    [avg(dot[132], dot[123]), dot[132], dot[215], dot[172], dot[136],
                     dot[169], dot[210], avg(dot[212], dot[202]), dot[57],
                     avg(dot[61], dot[186]),
                     dot[92], dot[206], dot[205], avg(dot[50], dot[147])]
                ))
            elif idx == 1:
                mask_list.append(np.array(
                    [avg(dot[361], dot[352]), dot[361], dot[435], dot[397], dot[365],
                     dot[394], dot[430], avg(dot[273], dot[287]), dot[287],
                     avg(dot[391], dot[410]),
                     dot[322], dot[426], dot[425], avg(dot[411], dot[280])]))
            elif idx == 2:
                mask_list.append(np.array(
                    [avg(dot[116], dot[93]), dot[116], dot[117], dot[118], dot[100],
                     avg(dot[126], dot[142]),
                     avg(dot[209], dot[142]), dot[49], dot[203], dot[205], dot[123]]
                ))
            elif idx == 3:
                mask_list.append(
                    np.array([avg(dot[345], dot[323]), dot[345], dot[346], dot[347], dot[329],
                              avg(dot[420], dot[371]),
                              avg(dot[371], dot[360]), dot[429], dot[423], dot[425], dot[352]]))
            elif idx == 4:
                mask_list.append(np.array(
                    [dot[61], dot[43], dot[204], dot[32], dot[171], dot[148], dot[176],
                     dot[149],
                     dot[150],
                     dot[169], dot[210], avg(dot[212], dot[202]),
                     avg(dot[57], avg(dot[57], dot[43]))]))
            elif idx == 5:
                mask_list.append(np.array(
                    [dot[291], dot[273], dot[424], dot[262], dot[396], dot[377], dot[400], dot[378],
                     dot[379],
                     dot[394], dot[430], avg(dot[432], dot[422]),
                     avg(dot[287], avg(dot[287], dot[273]))]))
            elif idx == 6:
                mask_list.append(np.array(
                    [dot[204], avg(dot[106], dot[91]), avg(dot[182], dot[181]), avg(dot[83], dot[84]),
                     avg(dot[18], dot[17]), avg(dot[314], dot[313]), avg(dot[405], dot[406]),
                     avg(dot[321], dot[335]), dot[424], dot[262], dot[396], dot[377], dot[152],
                     dot[148],
                     dot[171], dot[32]]))
            elif idx == 7:
                mask_list.append(np.array(
                    [dot[61], dot[92], avg(dot[206], dot[203]), dot[102], dot[48], dot[64], dot[98], dot[60],
                     dot[165]]))
            elif idx == 8:
                mask_list.append(np.array(
                    [dot[291], dot[322], avg(dot[426], dot[423]), dot[331], dot[294], dot[327], dot[290], dot[391]]))
            elif idx == 9:
                mask_list.append(np.array(
                    [dot[240], dot[64], dot[48], dot[131], avg(dot[134], avg(dot[134], dot[220])),
                     dot[45], avg(dot[4], avg(dot[4], dot[1])),
                     dot[275], avg(dot[363], avg(dot[363], dot[440])), dot[360], dot[278], dot[294],
                     dot[460], dot[305], dot[309], dot[438],
                     avg(dot[457], dot[275]), avg(dot[274], dot[275]), dot[274],
                     dot[458], dot[461], dot[462], dot[370],
                     dot[94], dot[141],
                     dot[242], dot[241], dot[238], dot[44],
                     avg(dot[45], dot[44]),
                     avg(dot[218], dot[45]), dot[218], dot[79], dot[75]]))
            elif idx == 10:
                mask_list.append(np.array(
                    [dot[60], dot[165], dot[61], dot[40], dot[39], dot[37], dot[97], dot[99]]))
            elif idx == 11:
                mask_list.append(np.array(
                    [dot[290], dot[391], dot[291], dot[270], dot[269], dot[267], dot[326],
                     dot[328]]))
            elif idx == 12:
                mask_list.append(np.array(
                    [dot[2], avg(dot[242], dot[97]), avg(dot[37], avg(dot[242], dot[37])), dot[0],
                     avg(dot[267], avg(dot[267], dot[462])), avg(dot[462], dot[326])]))
            elif idx == 13:
                mask_list.append(np.array([dot[4], avg(dot[45], avg(dot[45], dot[51])), dot[134],
                                           dot[51], dot[5], dot[281], dot[363],
                                           avg(dot[275], avg(dot[275], dot[281]))]))
            elif idx == 14:
                mask_list.append(np.array(
                    [dot[195], dot[248], dot[456], avg(dot[281], avg(dot[281], dot[248]))
                        , dot[5], avg(dot[51], avg(dot[51], dot[3])), dot[236], dot[3]]))
            elif idx == 15:
                mask_list.append(
                    np.array([dot[236], avg(dot[236], dot[134]),
                              avg(dot[198], dot[131]), avg(dot[126], dot[209]),
                              avg(dot[217], dot[198])])
                )
            elif idx == 16:
                mask_list.append(
                    np.array([dot[456], avg(dot[456], dot[363]),
                              avg(dot[420], dot[360]), avg(dot[355], dot[429]),
                              avg(dot[437], dot[420])])
                )
            elif idx == 17:
                mask_list.append(
                    np.array([dot[196], avg(dot[114], dot[217]), dot[236]])
                )
            elif idx == 18:
                mask_list.append(
                    np.array([dot[419], avg(dot[343], dot[437]), dot[456]])
                )
            elif idx == 19:
                mask_list.append(
                    np.array(
                        [avg(dot[114], dot[196]), dot[197], avg(dot[343], dot[419]), dot[351],
                         dot[417],
                         dot[285], dot[8],
                         dot[55], dot[193], dot[122]])
                )
            elif idx == 20:
                mask_list.append(
                    np.array(
                        [avg(dot[55], avg(dot[55], dot[8])), dot[8],
                         avg(dot[285], avg(dot[8], dot[285])),
                         dot[336],
                         avg(dot[337], avg(dot[337], dot[336])),
                         avg(dot[151], avg(dot[151], dot[9])),
                         avg(dot[108], avg(dot[108], dot[107])), dot[107]])
                )
            elif idx == 21:
                mask_list.append(
                    np.array(
                        [avg(dot[108], avg(dot[109], dot[69])),
                         avg(dot[104], avg(dot[68], dot[104])),
                         dot[105], dot[107]])
                )
            elif idx == 22:
                mask_list.append(
                    np.array(
                        [avg(dot[337], avg(dot[338], dot[299])),
                         avg(dot[333], avg(dot[298], dot[333])),
                         dot[334], dot[336]])
                )
            elif idx == 23:
                mask_list.append(
                    np.array(
                        [avg(dot[54], dot[103]), avg(dot[104], dot[68]), avg(dot[63], dot[105]),
                         dot[70],
                         dot[156], dot[143], dot[116], dot[234], dot[127],
                         dot[162], dot[21]])
                )
            elif idx == 24:
                mask_list.append(
                    np.array(
                        [avg(dot[332], dot[284]), avg(dot[298], dot[333]),
                         avg(dot[334], dot[293]),
                         dot[300], dot[383], dot[372], dot[345], dot[454], dot[356],
                         dot[389], dot[251]])
                )
            elif idx == 25:
                mask_list.append(
                    np.array(
                        [dot[53], dot[46], dot[156], dot[143], dot[111], dot[117], dot[118],
                         dot[119],
                         dot[120], dot[121],
                         dot[128], dot[245], dot[193], avg(dot[221], dot[55]), avg(dot[222], avg(dot[222], dot[65])),
                         dot[52]])
                )
            elif idx == 26:
                mask_list.append(
                    np.array(
                        [dot[283], dot[276], dot[383], dot[372], dot[340], dot[346], dot[347],
                         dot[348], dot[349], dot[350], dot[357], dot[465], dot[417], dot[441],
                         avg(dot[442], avg(dot[442], dot[295])), avg(dot[285], dot[441]), dot[282]]))
            elif idx == 27:
                mask_list.append(
                    np.array(
                        [
                            avg(dot[108], avg(dot[108], dot[109])),
                            avg(dot[151], avg(dot[10], avg(dot[151], avg(dot[151], dot[10])))),
                            avg(dot[337], avg(dot[337], dot[338])),
                            avg(dot[337], avg(dot[338], dot[299])),
                            avg(dot[337], avg(dot[337], dot[336])),
                            avg(dot[151], avg(dot[151], dot[9])),
                            avg(dot[108], avg(dot[108], dot[107])),
                            avg(dot[108], avg(dot[109], dot[69]))
                        ]
                    )
                )
            elif idx == 28:
                mask_list.append(
                    np.array(
                        [dot[103], dot[67], avg(dot[109], avg(dot[109], dot[67])),
                         avg(dot[108], avg(dot[108], dot[69])), dot[69], dot[104]])
                )
            elif idx == 29:
                mask_list.append(
                    np.array(
                        [dot[332], dot[297], avg(dot[338], avg(dot[338], dot[297])),
                         avg(dot[337], avg(dot[337], dot[299])), dot[299], dot[333]])
                )
            elif idx == 30:
                mask_list.append(
                    np.array(
                        [
                            avg(dot[108], avg(dot[109], dot[69])),
                            avg(dot[108], avg(dot[108], dot[109])),
                            avg(dot[151], avg(dot[10], avg(dot[151], avg(dot[151], dot[10])))),
                            avg(dot[337], avg(dot[337], dot[338])),
                            avg(dot[337], avg(dot[338], dot[299])), dot[338], dot[10], dot[109]
                        ]
                    ))

    return mask_list


def make_mask(dot):
    view_mask = []
    view_mask.append(np.array(
        [
            dot[152], dot[377], dot[400], dot[378], dot[379], dot[365], dot[397],
            dot[288], dot[301], dot[352], dot[447], dot[264], dot[389], dot[251],
            dot[284], dot[332], dot[297], dot[338], dot[10], dot[109], dot[67],
            dot[103], dot[54], dot[21], dot[162], dot[127], dot[234], dot[93],
            dot[132], dot[215], dot[58], dot[172], dot[136], dot[150], dot[149],
            dot[176], dot[148]
        ]
    ))
    remove_mask = []
    remove_mask.append(np.array(
        [
            dot[37], dot[39], dot[40], dot[185], dot[61], dot[57], dot[43], dot[106], dot[182], dot[83],
            dot[18], dot[313], dot[406], dot[335], dot[273], dot[287], dot[409], dot[270], dot[269],
            dot[267], dot[0], dot[37]
        ]
    ))
    remove_mask.append(np.array(
        [
            dot[37], dot[0], dot[267], dot[326], dot[2], dot[97], dot[37]
        ]
    ))
    remove_mask.append(np.array(
        [
            dot[2], dot[326], dot[327], dot[278], dot[279], dot[360], dot[363],
            dot[281], dot[5], dot[51], dot[134], dot[131], dot[49], dot[48],
            dot[98], dot[97], dot[2]
        ]
    ))
    remove_mask.append(np.array(
        [
            dot[236], dot[134], dot[51], dot[5], dot[281], dot[363], dot[456],
            dot[399], dot[412], dot[465], dot[413], dot[285], dot[336], dot[9],
            dot[107], dot[55], dot[189], dot[245], dot[188], dot[174], dot[236]
        ]
    ))
    remove_mask.append(np.array(
        [
            dot[336], dot[296], dot[334], dot[293], dot[283], dot[445], dot[342], dot[446],
            dot[261], dot[448], dot[449], dot[450], dot[451], dot[452], dot[453], dot[464],
            dot[413], dot[285], dot[336]
        ]
    ))
    remove_mask.append(np.array(
        [
            dot[107], dot[66], dot[105], dot[63], dot[53], dot[225], dot[113], dot[226],
            dot[31], dot[228], dot[229], dot[230], dot[231], dot[232], dot[233], dot[244],
            dot[189], dot[55], dot[107]
        ]
    ))

    return view_mask, remove_mask


def generate_maks(src, view, remove):
    shape = src.shape
    view_mask = np.zeros((shape[0], shape[1], 3), np.uint8)
    for (idx, mask) in enumerate(view):
        view_mask = cv2.fillConvexPoly(view_mask, mask.astype(int), color=(255, 255, 255))
    remove_mask = np.zeros((shape[0], shape[1], 3), np.uint8)
    for (idx, mask) in enumerate(remove):
        remove_mask = cv2.fillConvexPoly(remove_mask, mask.astype(int), color=(255, 255, 255))

    img = cv2.subtract(view_mask, remove_mask)

    rst = cv2.bitwise_and(src, img)

    return rst


def divide_array(data_array, count_array, frame_total):
    interval = max(count_array)
    rst = np.empty((frame_total, interval))
    i = 0
    cum_sum = 0
    for count in count_array:
        data = data_array[cum_sum:cum_sum + count]
        pad = interval - len(data)
        data = np.pad(data, (0, pad), 'constant', constant_values=0)
        rst[i] = data
        cum_sum += count
        i += 1

    return rst


def get_haarcascade():
    # haarcascade_url = "https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_alt2.xml"
    haarcascade_filename = "haarcascade_frontalface_alt2.xml"
    return cv2.CascadeClassifier(cv2.data.haarcascades + haarcascade_filename)


def get_eye_haarcascade():
    # haarcascade_url = "https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_alt2.xml"
    haarcascade_filename = "haarcascade_eye.xml"
    return cv2.CascadeClassifier(cv2.data.haarcascades + haarcascade_filename)


def get_frames_and_video_meta_data(video_path, meta_data_only=False):
    cap = cv2.VideoCapture(video_path)
    frameRate = cap.get(5)  # frame rate

    # Frame dimensions: WxH
    frame_dims = (int(cap.get(3)), int(cap.get(4)))
    # Paper mentions a stride of 0.5 seconds = 15 frames
    sliding_window_stride = int(frameRate / 2)
    num_frames = int(cap.get(7))
    if meta_data_only:
        return {"frame_rate": frameRate, "sliding_window_stride": sliding_window_stride, "num_frames": num_frames}

    # Frames from the video have shape NumFrames x H x W x C
    frames = np.zeros((num_frames, frame_dims[1], frame_dims[0], 3), dtype='uint8')

    frame_counter = 0
    while cap.isOpened():
        # curr_frame_id = int(cap.get(1))  # current frame number
        ret, frame = cap.read()
        if not ret:
            break

        frames[frame_counter, :, :, :] = frame
        frame_counter += 1
        if frame_counter == num_frames:
            break

    cap.release()
    return frames, frameRate, sliding_window_stride


def preprocess_video_to_st_maps(video_path, output_shape, clip_size=256):
    frames, frameRate, sliding_window_stride = get_frames_and_video_meta_data(video_path)
    num_frames = frames.shape[0]

    output_shape = (frames.shape[1], frames.shape[2])
    num_maps = int((num_frames - clip_size) / sliding_window_stride + 1)
    print(video_path + "  " + str(num_frames) + "  " + str(num_maps) + "  " + str(clip_size) + "  " + str(
        sliding_window_stride))
    if num_maps < 0:
        # print(num_maps)
        print(video_path)
        return None

    # stacked_maps is the all the st maps for a given video (=num_maps) stacked.
    stacked_ptts = np.zeros((num_maps, 5 * 64, frames.shape[2], 3))
    stacked_maps = np.zeros((num_maps, 64, clip_size, 3))

    # processed_maps will contain all the data after processing each frame, but not yet converted into maps
    processed_maps = np.zeros((num_frames, 25, 3))
    processed_ptts = np.zeros((frames.shape[2], 25, 3))
    processed_frames = np.zeros((num_frames, output_shape[1], output_shape[0], 3))
    # processed_frames = []
    map_index = 0

    # Init scaler and detector
    min_max_scaler = preprocessing.MinMaxScaler()
    detector = get_haarcascade()
    eye_detector = get_eye_haarcascade()

    # First we process all the frames and then work with sliding window to save repeated processing for the same frame index
    for idx, frame in enumerate(frames):
        # spatio_temporal_map = np.zeros((fr, 25, 3))
        '''
           Preprocess the Image
           Step 1: Use cv2 face detector based on Haar cascades
           Step 2: Crop the frame based on the face co-ordinates (we need to do 160%)
           Step 3: Downsample the face cropped frame to output_shape = 36x36
       '''
        faces = detector.detectMultiScale(frame, 1.3, 5)
        if len(faces) is not 0:
            (x, y, w, d) = faces[0]
            frame_cropped = frame[y:(y + d), x:(x + w)]
            frame_masked = frame_cropped
        else:
            mask = np.zeros(frame.shape[:2], dtype="uint8")
            frame_masked = cv2.bitwise_and(frame, frame, mask=mask)

        try:
            frame_resized = cv2.resize(frame_masked, output_shape, interpolation=cv2.INTER_CUBIC)
            frame_resized = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2YUV)

        except:
            print('\n--------- ERROR! -----------\nUsual cv empty error')
            print(f'Shape of img1: {frame.shape}')
            # print(f'bbox: {bbox}')
            print(f'This is at idx: {idx}')
            exit(666)

        processed_frames[idx, :, :, :] = frame_resized

    # At this point we have the processed maps from all the frames in a video and now we do the sliding window part.
    for start_frame_index in range(0, num_frames, sliding_window_stride):
        end_frame_index = start_frame_index + clip_size
        if end_frame_index > num_frames:
            break
        spatio_temporal_map = np.zeros((clip_size, 64, 3))

        for idx, frame in enumerate(processed_frames[start_frame_index:end_frame_index]):
            roi_blocks = chunkify(frame, 8, 8)
            for block_idx, block in enumerate(roi_blocks):
                avg_pixels = cv2.mean(block)
                spatio_temporal_map[idx, block_idx, 0] = avg_pixels[0]
                spatio_temporal_map[idx, block_idx, 1] = avg_pixels[1]
                spatio_temporal_map[idx, block_idx, 2] = avg_pixels[2]

        for block_idx in range(spatio_temporal_map.shape[1]):
            # Not sure about uint8
            fn_scale_0_255 = lambda x: (x * 255.0).astype(np.uint8)
            scaled_channel_0 = min_max_scaler.fit_transform(spatio_temporal_map[:, block_idx, 0].reshape(-1, 1))
            spatio_temporal_map[:, block_idx, 0] = fn_scale_0_255(scaled_channel_0.flatten())
            scaled_channel_1 = min_max_scaler.fit_transform(spatio_temporal_map[:, block_idx, 1].reshape(-1, 1))
            spatio_temporal_map[:, block_idx, 1] = fn_scale_0_255(scaled_channel_1.flatten())
            scaled_channel_2 = min_max_scaler.fit_transform(spatio_temporal_map[:, block_idx, 2].reshape(-1, 1))
            spatio_temporal_map[:, block_idx, 2] = fn_scale_0_255(scaled_channel_2.flatten())

        stacked_maps[map_index, :, :, :] = np.transpose(spatio_temporal_map, (1, 0, 2))

        transpose = np.transpose(processed_frames[start_frame_index:end_frame_index], (1, 0, 2, 3))

        ptt_map = np.zeros((np.shape(transpose)[0], 5 * 64, 3))
        for idx, frame in enumerate(transpose):
            roi_blocks = chunkify(frame, block_width=64, block_height=5)
            for block_idx, block in enumerate(roi_blocks):
                avg_pixels = cv2.mean(block)
                ptt_map[idx, block_idx, 0] = avg_pixels[0]
                ptt_map[idx, block_idx, 1] = avg_pixels[1]
                ptt_map[idx, block_idx, 2] = avg_pixels[2]

        for block_idx in range(ptt_map.shape[1]):
            # Not sure about uint8
            fn_scale_0_255 = lambda x: (x * 255.0).astype(np.uint8)
            scaled_channel_0 = min_max_scaler.fit_transform(ptt_map[:, block_idx, 0].reshape(-1, 1))
            ptt_map[:, block_idx, 0] = fn_scale_0_255(scaled_channel_0.flatten())
            scaled_channel_1 = min_max_scaler.fit_transform(ptt_map[:, block_idx, 1].reshape(-1, 1))
            ptt_map[:, block_idx, 1] = fn_scale_0_255(scaled_channel_1.flatten())
            scaled_channel_2 = min_max_scaler.fit_transform(ptt_map[:, block_idx, 2].reshape(-1, 1))
            ptt_map[:, block_idx, 2] = fn_scale_0_255(scaled_channel_2.flatten())
        stacked_ptts[map_index, :, :, :] = np.transpose(ptt_map, (1, 0, 2))

        map_index += 1

    return stacked_maps, sliding_window_stride, num_frames, stacked_ptts
    # rst,bvp,sliding,frames,ptt
    # return 1,1,num_frames,1


def chunkify(img, block_width=5, block_height=5):
    shape = img.shape
    x_len = shape[0] // block_width
    y_len = shape[1] // block_height
    # print(x_len, y_len)

    chunks = []
    x_indices = [i for i in range(0, shape[0] + 1, x_len)]
    y_indices = [i for i in range(0, shape[1] + 1, y_len)]

    shapes = list(zip(x_indices, y_indices))

    #  # for plotting purpose
    # implot = plt.imshow(img)
    #
    # end_x_list = []
    # end_y_list = []

    for i in range(len(x_indices) - 1):
        # try:
        start_x = x_indices[i]
        end_x = x_indices[i + 1]
        for j in range(len(y_indices) - 1):
            start_y = y_indices[j]
            end_y = y_indices[j + 1]
            # end_x_list.append(end_x)
            # end_y_list.append(end_y)
            chunks.append(img[start_x:end_x, start_y:end_y])
        # except IndexError:
        #     print('End of Array')

    return chunks


def preprocess_video(video_path, output_shape, clip_size=256):
    cap = cv2.VideoCapture(video_path)
    frameRate = cap.get(5)  # frame rate

    # Frame dimensions: WxH
    frame_dims = (int(cap.get(3)), int(cap.get(4)))
    # Paper mentions a stride of 0.5 seconds = 15 frames
    sliding_window_stride = int(frameRate / 2)
    num_frames = int(cap.get(7))
    # output_shape = (frames.shape[1], frames.shape[2])
    output_shape = (clip_size, clip_size)
    num_maps = int((num_frames - clip_size) / sliding_window_stride + 1)
    if num_maps < 0:
        # print(num_maps)
        print(video_path)
        return None

    # stacked_maps is the all the st maps for a given video (=num_maps) stacked.
    stacked_maps = np.zeros((num_maps, clip_size, clip_size, clip_size, 3))
    # processed_maps will contain all the data after processing each frame, but not yet converted into maps
    # processed_maps = np.zeros((num_frames, 25, 3))
    # processed_frames = np.zeros((num_frames, output_shape[0], output_shape[1], 3))
    processed_frames = []
    map_index = 0

    # Init scaler and detector
    min_max_scaler = preprocessing.MinMaxScaler()
    detector = get_haarcascade()
    eye_detector = get_eye_haarcascade()

    # resized_frame = np.empty((num_frames,clip_size,clip_size,3))
    with tqdm(total=num_frames, position=0, leave=True, desc=video_path) as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if frame is None:
                break
            faces = detector.detectMultiScale(frame, 1.3, 5)
            if len(faces) is not 0:
                (x, y, w, d) = faces[0]
                frame_cropped = frame[y:(y + d), x:(x + w)]
                # eyes = eye_detector.detectMultiScale(frame_cropped, 1.2, 3)
                frame_masked = frame_cropped
            else:
                # The problemis that this doesn't get cropped :/
                # (x, y, w, d) = (308, 189, 215, 215)
                # frame_masked = frame[y:(y + d), x:(x + w)]

                # print("face detection failed, image frame will be masked")
                mask = np.zeros(frame.shape[:2], dtype="uint8")
                frame_masked = cv2.bitwise_and(frame, frame, mask=mask)
            try:
                frame_resized = cv2.resize(frame_masked, output_shape, interpolation=cv2.INTER_CUBIC)
                frame_resized = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2YUV)

            except:
                print('\n--------- ERROR! -----------\nUsual cv empty error')
                print(f'Shape of img1: {frame.shape}')
                # print(f'bbox: {bbox}')
                # print(f'This is at idx: {idx}')
                exit(666)
            processed_frames.append(frame_resized)
            pbar.update(1)
    cap.release()

    # At this point we have the processed maps from all the frames in a video and now we do the sliding window part.
    for start_frame_index in range(0, num_frames, sliding_window_stride):
        end_frame_index = start_frame_index + clip_size
        if end_frame_index > num_frames:
            break
        stacked_maps[map_index, :, :, :, :] = np.asarray(processed_frames[start_frame_index:end_frame_index])
        map_index += 1

    return stacked_maps, num_maps, num_frames
