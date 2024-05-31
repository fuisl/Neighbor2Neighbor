"""
This module is use to calculate the metric of the output.
"""

import cv2
import numpy as np

class Metric:
    def __init__(self):
        pass

class CNR(Metric):
    def __init__(self, img):
        self.img = img
        self.rois = []
        self.bg_roi = None

    def cal(self):
        """
        Choose the ROIs and calculate the CNR.
        """
        # Choose background ROI
        bg_roi = cv2.selectROI('Background ROI', self.img, showCrosshair=False, printNotice=False)
        x, y, w, h = bg_roi
        self.bg_roi = self.img[y:y+h, x:x+w]
        cv2.rectangle(self.img, (x, y), (x+w, y+h), (0, 0, 255), 1)
        cv2.destroyAllWindows()

        # Choose the ROIs
        while True:
            roi = cv2.selectROI('Select ROI', self.img, showCrosshair=False, printNotice=False)
            x, y, w, h = roi
            self.rois.append(self.img[y:y+h, x:x+w])
            cnr = self.calculateCNR(self.rois[-1])
            cv2.rectangle(self.img, (x, y), (x+w, y+h), (0, 255, 0), 1)
            cv2.putText(self.img, f'{cnr:.2f}', (x, y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1)

            if cv2.waitKey(0) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()        

    def calculateCNR(self, roi):
        bg_mean = np.mean(self.bg_roi)
        bg_std = np.std(self.bg_roi)

        roi_mean = np.mean(roi)
        roi_std = np.std(roi)

        cnr = 10 * np.log((np.abs(roi_mean - bg_mean)) / np.sqrt(0.5*(roi_std**2) + (bg_std**2)))

        return cnr


class MSR(Metric):
    def __init__(self, img):
        self.img = img
        self.rois = []

    def cal(self):
        """Choose the ROI and calculate the MSR."""

        while True:
            roi = cv2.selectROI('Select ROI', self.img, showCrosshair=False, printNotice=False)
            x, y, w, h = roi
            self.rois.append(self.img[y:y+h, x:x+w])
            msr = self.calculateMSR(self.rois[-1])
            cv2.rectangle(self.img, (x, y), (x+w, y+h), (0, 255, 0), 1)
            cv2.putText(self.img, f'{msr:.2f}', (x, y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1)

            if cv2.waitKey(0) & 0xFF == ord('q'):
                break

    def calculateMSR(self, roi):
        roi_mean = np.mean(roi)
        roi_std = np.std(roi)

        return roi_mean / roi_std
        
class EP(Metric):
    def __init__(self, img):
        self.img = img
        self.rois = []

class TP(Metric):
    def __init__(self, img_in, img_den):
        self.img_in = img_in
        self.img_den = img_den
        self.rois = []

    def cal(self):
        """Choose the ROI and calculate the TP."""

        while True:
            roi = cv2.selectROI('Select ROI', self.img_den, showCrosshair=False, printNotice=False)
            x, y, w, h = roi

            roi_in = self.img_in[y:y+h, x:x+w]
            roi_den = self.img_den[y:y+h, x:x+w]

            self.rois.append((roi_in, roi_den))
            tp = self.calculateTP(roi_in, roi_den)
            cv2.rectangle(self.img_den, (x, y), (x+w, y+h), (0, 255, 0), 1)
            cv2.putText(self.img_den, f'{tp:.2f}', (x, y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1)

            if cv2.waitKey(0) & 0xFF == ord('q'):
                break


    def calculateTP(self, roi_in, roi_den):
        noisy_std = np.std(roi_in)
        denoised_std = np.std(roi_den)

        noisy_mean = np.mean(roi_in)
        denoised_mean = np.mean(roi_den)
        
        tp = (denoised_std**2 / noisy_std**2) * np.sqrt(denoised_mean / noisy_mean)

        return tp
    
class EP(Metric):
    def __init__(self, img_in, img_den):
        self.img_in_prime = img_in
        self.img_den = img_den
        self.rois = []

    def cal():
        pass

    def calculateEP(self, roi_in_prime, roi_den):
        delta_prime = cv2.Laplacian(roi_in_prime, cv2.CV_64F)
        delta_den = cv2.Laplacian(roi_den, cv2.CV_64F)

        