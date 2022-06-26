# Notebooks v5

Start working on denbi and initialize project meant for denoising. 


## 5.00 get dummy reconstruction data

First we need to get some 3d images with no missing wedge to use as reference. Then apply the missing wedge and after the denoising is done, we can compare the images with the ground truth.

31.05 It might be the case that no reconstruction can be done if there is not sufficient information missing, thus I decided to increase the angular extent for the Lung dummies to see if Isonet might improve the original data. I did this after reviewing the results and looking at one prediction with better SSIM than the corrupted input.

## 5.01 visualize isoNet results

Looks like even when IsoNet recovers the missing wedge, the SSIM and NMI values are smaller when using the corrected tomograms wrt GT compared to the corrupted values. Could it be that the pixel size is affecting the reconstruction? I think this should only be a problem if we were running the deconvolution.

This behaviour is similar in both dummy datasets. Maybe there is an ingredient missing in isoNet.

02.06 Increasing the MW angular extent doesn't seem to affect the reconstruction capability of the model. The results seem still quite bad (Lung dummy ae31)

05.06 It seems that adding several versions of the rotated arrays do not improve isonet's performance (brain_dummy rotated)

## 5.02 EDA Fourier

For both the spinach membranes and salmonella minicells, setting to zero the low frequencies withing some cube doesn't seem to have a very large effect on image deterioration. Thus, it seems that most of the important information for the tomogram is on the high frequency components. 

For some reason, it seems that most of the information in both of the tomograms I saw (tomo32 and 20200213_GW214_tgt4_ali) is condensed in 20º region. This observation arises from taking a missing X of 10 degrees per arm at different degrees from the X-axis. When the cross starts at 20º, most of the tomogram's information is lost. What is special about the 20º?

In our case, the distribution of Fourier coefficients within and annulus does not seem to follow a Rayleigh distribution. The question remains if there is some kind of spectral consistency we could exploit in e.g. a Graph NN.

## 5.04 EDA Fourier 2

Make shell and make ring functions.

It turns out that running the FFT on XZ slices yields a spectrum where the missing wedge can be seen (why?). Thus we can just work on 2D Images for now.

## 5.05 Neighbor selection

I think we could just use low frequency parts of the FFT spectrum to reconstruct the images and work on this smaller set. The SSIM is almost the same when filtering frequencies outside a radius of 200 and filtering low power.


# Notes

ICON paper: "Since specimens are illuminated by parallel beam, projections along planes perpendicular to the tilt axis are mathematically independent." Thus, maybe including 3D information in Fourier space does not necessarily help in any way. 