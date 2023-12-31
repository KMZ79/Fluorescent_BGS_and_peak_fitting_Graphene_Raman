## Useage
Code written for use in PhD (Studies of Substrate Mediated Modification of Chemical Vapour Deposition Graphene, Khadisha M. Zahra, University of Manchester, 2023) and will not be maintained. 

Handles txt files and was created to analyse Renishaw data on PCs other than the instrument PC due to licsence restrictions. 

Heavily commented with directions for use. 

Subtraction of the photoluminescent background caused by the Cu substrate is applied to all graphene/Cu sample datasets. 
This is done by masking the D, G and 2D peaks and fitting the remaining data with a polynomial (generally set to 6th order but can be changed). 
Using this fit, the background was then removed and Lorentzian peaks were fitted to determine their FWHM, peak intensity and peak position. 
It also uses this data to calculate intensity ratios. 
Contour maps and histograms were created to visualise the results.

Can also remove cosmic rays

Can handle single spectra and Raman maps.

The research made use of the following Python packages: NumPy, Matplotlib, AstroPy.

