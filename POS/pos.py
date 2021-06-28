import matplotlib
#matplotlib.use("TkAgg")
from SkinDetector.skin_detector import skin_detector

from matplotlib import pyplot as plt

import os
import sys
sys.path.insert(0, './SkinDetector')
import pkg_resources

import numpy as np
import cv2
import dlib

import argparse

def main(video_path):
    # EXTRACT PULSE
    start = 0
    end = 450

    framerate = 30

    # FREQUENCY ANALYSIS
    nsegments = 12

    plot =  False
    image_show = True

    left_increase_ratio = 0.05 #5%
    top_increase_ratio = 0.25 #5%

    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", default = video_path, help = "path to the (optional) video file")
    args = vars(ap.parse_args())


    if not args.get("video", False):
        from_webcam = True
        camera = cv2.VideoCapture(0)
        start = 0
        end = 450
	# otherwise, load the video
    else:
        camera = cv2.VideoCapture(args["video"])

    video_file_path = args["video"]
    video_file_name = os.path.basename(video_file_path)

    start_index = start
    end_index = end

    # number of final frames
    if end_index > 0:
        nb_frames = end_index - start_index

    # loop on video frames
    frame_counter = 0
    i = start_index

    detector = dlib.get_frontal_face_detector()

    while (i >= start_index and i < end_index):
        (grabbed, frame) = camera.read()

        if not grabbed:
            continue

        h,w,_ = frame.shape

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)

        if len(rects)==0:
            continue

        if image_show:
            show_frame = frame.copy()

        if(len(rects)>0):
            rect = rects[0]

            left, right, top, bottom = rect.left(), rect.right(), rect.top(),rect.bottom()
            width = abs(right - left)
            height = abs(bottom - top)
            #print("Width and Height of bounding box : ",width,height)

            face_left = int(left - (left_increase_ratio/2)*width)
            face_top = int(top - (top_increase_ratio)*height)
            #face_right = int(right + (area_increase_ratio/2)*width)
            #face_bottom = int(bottom + (area_increase_ratio/2)*height)

            face_right = right
            face_bottom = bottom

            ##print("Increased coordinates: ",face_left, face_right, face_top, face_bottom)

            if image_show:
                cv2.rectangle(show_frame,(left,top),(right,bottom),(255,255,0),3)
                cv2.rectangle(show_frame,(face_left,face_top),(face_right,face_bottom),(0,255,0),3)

            face = frame[face_top:face_bottom,face_left:face_right]

            if(face.size==0):
                continue
            #    continue
            #Extract face skin pixels
            mask = skin_detector.process(face)

            #print("Mask shape: ",mask.shape)
            masked_face = cv2.bitwise_and(face, face, mask=mask)
            number_of_skin_pixels = np.sum(mask>0)

            #compute mean
            r = np.sum(masked_face[:,:,2])/number_of_skin_pixels
            g = np.sum(masked_face[:,:,1])/number_of_skin_pixels
            b = np.sum(masked_face[:,:,0])/number_of_skin_pixels

            if frame_counter==0:
                mean_rgb = np.array([r,g,b])
            else:
                mean_rgb = np.vstack((mean_rgb,np.array([r,g,b])))

            ##print("Mean RGB -> R = {0}, G = {1}, B = {2} ".format(r,g,b))

        if image_show:
            if h>w and h>640:
                    dim = (int(640 * (w/h)),640)
                    show_frame = cv2.resize(show_frame, dim, interpolation = cv2.INTER_LINEAR)
            if w>h and w>640:
                    dim = (640, int(640 * (h/w)))
                    show_frame = cv2.resize(show_frame, dim, interpolation = cv2.INTER_LINEAR)

        sub_name = video_path.split('/')
        sub = sub_name[-2]

        #cv2.imshow("frame",show_frame)
        if(image_show):
            cv2.imshow( sub, masked_face)
            cv2.waitKey(1)
        frame_counter +=1
        i += 1
        #end loop

    camera.release()
    cv2.destroyAllWindows()

    # Calculating window length(l)
    #          - 1.6s : can capture at least one cardiac cycle in [40;240]bpm
    l = int(framerate * 1.6) # 32

    # Initialize
    H = np.zeros(mean_rgb.shape[0])

    for n in range(mean_rgb.shape[0]-l) :

        # Step1 :Spatial Averaging
        C = mean_rgb[n:n+l-1,:].T # Sliding each window

        # Step2 :Temporal normalization - Cn=diag(mean(C,2))^-1*C
        mean_C = np.mean(C, axis=1) # Mean
        diag_mean_C = np.diag(mean_C) # Diagonal
        diag_mean_C_inv = np.linalg.inv(diag_mean_C) # Inverse
        Cn = np.matmul(diag_mean_C_inv, C)

        # Step3 :Projection(3D signal to 2D signal)
        projection_matrix = np.array([[0,1,-1],[-2,1,1]])
        S = np.matmul(projection_matrix, Cn)

        #Step4 :Tuning(2D signal to 1D signal)
        # - when the pulsatile variation dominates S(t), S1(t) and S2(t) appear in in-phase.
        std = np.array([1, np.std(S[0, :]) / np.std(S[1, :])])
        P = np.matmul(std, S)

        #Step5 :Overlap-adding
        H[n:n+l-1] = H[n:n+l-1] + (P-np.mean(P))/np.std(P)

    print("Pulse",H)
    signal = H
    print("Pulse shape", H.shape)

    #FFT to find the maxiumum frequency
    # find the segment length, such that we have 8 50% overlapping segments (Matlab's default)
    #segment_length = (2*signal.shape[0]) // (nsegments + 1)
    #print("nperseg",segment_length)

    from matplotlib import pyplot
    pyplot.plot(range(signal.shape[0]), signal, 'g')
    pyplot.title('Filtered green signal')
    pyplot.show()

    # Transfer to frequency domain to find ROI
    from scipy.signal import welch
    signal = signal.flatten()
    green_f, green_psd = welch(signal, framerate, 'flattop', nperseg=300)
    print("Green F, Shape",green_f,green_f.shape)
    print("Green PSD, Shape",green_psd,green_psd.shape)

    #green_psd = green_psd.flatten()
    # Set the range of target freqency range
    first = np.where(green_f > 0.9)[0] #0.8 for 300 frames
    last = np.where(green_f < 2.2)[0]
    first_index = first[0]
    last_index = last[-1]
    range_of_interest = range(first_index, last_index + 1, 1)

    print("Range of interest",range_of_interest)
    max_idx = np.argmax(green_psd[range_of_interest])
    f_max = green_f[range_of_interest[max_idx]]

    hr = f_max*60.0
    print("Heart rate = {0}".format(hr))

    from matplotlib import pyplot
    pyplot.semilogy(green_f, green_psd, 'g')
    xmax, xmin, ymax, ymin = pyplot.axis()
    pyplot.vlines(green_f[range_of_interest[max_idx]], ymin, ymax, color='red')
    pyplot.title('Power spectrum of the green signal (HR = {0:.1f})'.format(hr))
    pyplot.show()

    return hr


if __name__ == "__main__":

    # Select data_except 15, 16, 25, 43,49
    file_list = [1,3,4,8,9,10,11,12,13,14,17,20,22,23,24,26,27,30,32,31,33,34,35,36,37,38,39,40,41,42,44,45,46,47,48]
    files = list(map(str, file_list))
    value=[]
    for i in files:
        path = '/mnt/a7930c08-d429-42fa-a09e-15291e166a27/BVP_js/subject'+i+'/vid.avi'
        print("PROCESSING-----------------------------------------",i,"/",len(file_list))
        hr=main(path)

        # Save result as a txt file
        f= open('./Result_HR.txt', 'a')
        f.write(str(hr))
        f.write("\n")
    f.close()