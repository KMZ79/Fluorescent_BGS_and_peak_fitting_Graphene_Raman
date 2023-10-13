## Useage
Code written for use in PhD (Studies of Substrate Mediated Modification of Chemical Vapour Deposition Graphene, Khadisha M. Zahra, University of Manchester 2023) and will not be maintained. 

Subtraction of the photoluminescent background caused by the Cu substrate iss applied to all graphene/Cu sample datasets. 
This is done by masking the D, G and 2D peaks and fitting the remaining data with a 6th order polynomial. 
Using this fit, the background was then removed and Lorentzian peaks were fitted to determine their FWHM, peak intensity and peak position. 
Also uses this data to calculate intensity ratios. 
Contour maps and histograms were created to visualise the results.
Can also remove cosmic rays
Can handle single spectra and Raman maps.
Was created to analyse Renishaw data on PCs other than the instrument PC due to licsence restrictions. 
The research made use of the following Python packages: NumPy, Matplotlib, AstroPy.
Heavily commented with directions for use. 
