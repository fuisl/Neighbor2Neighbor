import numpy as np
import cv2
from PIL import Image

class Metric:
    """
    This class is used to calculate the metric of the output from the denoised model

    Parameters:

    :param noisy: The noisy image
    :type noisy: numpy.ndarray

    :param denoised: The denoised image
    :type denoised: numpy.ndarray
    """
    def __init__(self, noisy, denoised):
        self.noisy = noisy
        self.denoised = denoised
        self.rois = []
        self.bg_roi = None

        # Init default rois
        self.bg_roi = (200, 0, 50, 25)  # x, y, w, h

        self.rois.append((0, 0, 25, 25))
        self.rois.append((0, 100, 25, 25))
        self.rois.append((150, 125, 50, 25))
        self.rois.append((200, 100, 50, 25))
        self.rois.append((50, 125, 25, 25))
        self.rois.append((100, 175, 35, 25))
        self.rois.append((200, 150, 35, 25))

    def chooose_roi(self):
        """
        Choose the ROIs for all metric calculation with ROIs required.
        """
        pass

    def calculate_psnr(self):
        """
        Calculate the PSNR of the denoised image
        """
        diff = self.noisy - self.denoised
        mse = np.mean(diff**2)
        psnr = 10 * np.log10(255**2 / mse)
        return psnr
    
    # def calculate_ssim(self):
    #     """
    #     Calculate the SSIM of the denoised image
    #     """
    #     ssim = cv2.SSIM(self.noisy, self.denoised)
    #     return ssim
    
    def calculate_cnr(self, denoised=True):
        """
        Calculate the CNR of the denoised image
        """
        if denoised:
            img = self.denoised
        else:
            img = self.noisy
        
        x, y, w, h = self.bg_roi
        extracted_bg = img[y:y+h, x:x+w]  
        
        cnrs = []

        for roi in self.rois:
            img_roi = img[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]]
            bg_mean = np.mean(extracted_bg)
            bg_std = np.std(extracted_bg)
            roi_mean = np.mean(img_roi)
            roi_std = np.std(img_roi)

            cnr = 10 * np.log((np.abs(roi_mean - bg_mean)) / np.sqrt(0.5*(roi_std**2) + (bg_std**2)))
            cnrs.append(cnr)

        return cnrs

    def calculate_msr(self):
        """
        Calculate the MSR of the denoised image
        """
        msrs = []

        for roi in self.rois:
            img_roi = self.denoised[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]]
            
            roi_mean = np.mean(img_roi)
            roi_std = np.std(img_roi)

            msr = roi_mean / roi_std
            msrs.append(msr)

        return msrs

    def calculate_tp(self):
        """
        Calculate the TP of the denoised image vs the noisy image
        """
        tps = []
        
        for roi in self.rois:
            roi_noisy = self.noisy[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]]
            roi_denoised = self.denoised[roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]]

            noisy_std = np.std(roi_noisy)
            denoised_std = np.std(roi_denoised)

            noisy_mean = np.mean(roi_noisy)
            denoised_mean = np.mean(roi_denoised)

            tp = (denoised_std**2 / noisy_std**2) * np.sqrt(denoised_mean / noisy_mean)

            tps.append(tp)

        return tps

        return tps

    def calculate_ep(self):
        """
        Calculate the EP of the denoised image
        """
        pass

    def calculate_metrics(self):
        """
        Calculate all the metrics
        """
        psnr = self.calculate_psnr()
        # ssim = self.calculate_ssim()
        cnr = self.calculate_cnr()
        msr = self.calculate_msr()
        tp = self.calculate_tp()

        return {'psnr': psnr, 'cnr': cnr, 'msr': msr, 'tp': tp}
