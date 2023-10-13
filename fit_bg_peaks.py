    import numpy as np ## Mostly my data array handling and manip
    import matplotlib.pyplot as plt ## Figures and plots
    import scipy ## Scienitific Python Library (not using)
    from scipy.signal import find_peaks ## Using to detect cosmic rays
    from astropy.modeling import models, fitting ## Is doing my Lorentzian fitting model
    
    from tkinter import filedialog ## Opens window to select and read file
    from tkinter import *
    ######################################################################################
    
    ## RAMAN GRAPHENE (BG remove)
    D_RangeBG = [1300, 1400] 
    G_RangeBG = [1510, 1640]
    GPrime_RangeBG = [2420,2500]
    TwoD_RangeBG =[2610, 2750]
    
    ## RAMAN GRAPHENE (Peak (LORENTZ) Fit - slightly wider)
    D_RangePF = [1250, 1450] 
    G_RangePF = [1450, 1750]
    GPrime_RangePF = [2300,2650]
    TwoD_RangePF =[2550, 2800]
    ######################################################################################
    
    ## RAMAN VAPOUR (BG remove)
    Oxi_RangeBG = [1525, 1575]
    Nitr_RangeBG = [2305, 2355]
    
    ## RAMAN VAPOUR (Peak (LORENTZ) Fit - slightly wider)
    Oxi_RangePF = [1500, 1600]
    Nitr_RangePF = [2275, 2375]
    ######################################################################################
    
    ### FUNCTIONS : SPECTRA HANDLING 
    def isolate_xy_spectra(coordinates, data):
    
        ## Set desired x, y into an array
        ## np.array - creates an array (i.e. of a singular xy coord)
        wantcoords = np.array([coordinates[:2]])
    
        ## Create a Boolean array for each column (x, y)
        ## == checks for equality (i.e. from original data finds spectra with given coords)
        useindicesx = data[:,0]==wantcoords[:, 0]
        useindicesy = data[:,1]==wantcoords[:, 1]
    
        ## Multiplication of the 2 Booleans (i.e. F*F=F, F*T=F, T*T=T) (same size of the original)
        usetheseTF = useindicesx * useindicesy
    
        ## Use T/F array (that shares an axis) like indices creating a shorter array 
        ## of only the corresponding T value from the Boolean
        datasubset = data[usetheseTF, 2:]
    
        #print(datasubset)
        return (datasubset)
    ######################################################################################
    
    ### FUNCTIONS : SPECTRA HANDLING
    def data_range(datasubset, range):
    
        ## Creating Boolean array from specified ranges for each peak [see @Top Raman ranges]
        ## np.logical_and - computes the truth value of x1 AND x2
        boolean_array = np.logical_and(datasubset[:,0] > range[0], datasubset[:,0] < range[1])
        
        return(boolean_array)
    ######################################################################################
        
    ### FUNCTION : BACKGROUND REMOVAL
    def remove_bg_from_raw(spectrum, degoffit=6):
    
            ## Creating Boolean array for each GRAPHENE (Gr) peak [see @Top Raman ranges]
            inpeakD_BG = data_range(spectrum, D_RangeBG)
            inpeakG_BG = data_range(spectrum, G_RangeBG)
            inpeakGPrime_BG = data_range(spectrum, GPrime_RangeBG)
            inpeakTwoD_BG = data_range(spectrum, TwoD_RangeBG)
    
            ## Selecting Data for Gr peaks
            Data_peakD_BG = spectrum[inpeakD_BG, :]
            Data_peakG_BG = spectrum[inpeakG_BG, :]
            Data_peakGPrime_BG = spectrum[inpeakGPrime_BG, :]
            Data_peakTwoD_BG = spectrum[inpeakTwoD_BG, :]
    
            ## Creating Boolean array for each VAPOUR peak [see @Top Raman ranges]
            inpeakOxi_BG = data_range(spectrum, Oxi_RangeBG)
            inpeakNitr_BG = data_range(spectrum, Nitr_RangeBG)
    
            ## Selecting Data for Vapour peaks
            Data_peakOxi_BG = spectrum[inpeakOxi_BG, :]
            Data_peakNitr_BG = spectrum[inpeakNitr_BG, :]
    ######################################################################################    
    
            ## np.logical_or - combines the ranges for each specified Gr peak (D, G, G', 2D)
            inallpeaks = np.logical_or(np.logical_or(inpeakD_BG, inpeakG_BG), np.logical_or(inpeakTwoD_BG, inpeakGPrime_BG))
    
            ## ~ is specifying all data that is NOT in the 'inallpeaks' ranges
            subset_not_peaks = spectrum[~inallpeaks]
           
    
            ## np.poly - finds the coeff. or a poly with given sequence roots [see DoF specified by user input or pre-select]
            baseline_poly = np.polyfit(subset_not_peaks[:,0], subset_not_peaks[:,1], degoffit)
            #print("baseline_poly", baseline_poly)
            
    
            ## Calculates the values of intensity (i.e. y) for the given DoF and Coef
            ## np.zeros - returns new array of given shape and type but filled with zeros
            baseline_poly_vals = np.polyval(baseline_poly, spectrum[:,0])
            #baseline_poly_vals = np.zeros(spectrum[:,0].shape)
            #for coeff in range(degoffit+1):
            #    baseline_poly_vals += baseline_poly[coeff] * spectrum[:,0] ** (degoffit-coeff)
            
            ## FIGURE :         
            plt.figure()
            plt.plot(spectrum[:, 0], spectrum[:, 1])
            plt.plot(spectrum[:,0], baseline_poly_vals)
            plt.gca().axvspan(D_RangeBG[0], D_RangeBG[1], color='0.7', alpha=0.5)
            plt.gca().axvspan(G_RangeBG[0], G_RangeBG[1], color='0.7', alpha=0.5)
            plt.gca().axvspan(GPrime_RangeBG[0], GPrime_RangeBG[1], color='0.7', alpha=0.5)
            plt.gca().axvspan(TwoD_RangeBG[0], TwoD_RangeBG[1], color='0.7', alpha=0.5)
            plt.title("Raw Data With Polynomial Fit at xy")
            plt.xlabel("Wavenumber (cm-1)")
            plt.ylabel("Intensity (Arb.)")
            plt.show()
            
            baseline_subtract = spectrum[:,1] - baseline_poly_vals
            
            ## Calculate the signal-to-noise of the background
            cosmic_candidates = baseline_subtract > (np.mean(baseline_subtract) + 10*np.std(baseline_subtract))
            background_STD = np.std(baseline_subtract[~cosmic_candidates])
            
            ## FIGURE : Baseline subtracted data
            #plt.figure()
            #plt.plot(spectrum[:,0], baseline_subtract)
            #plt.show()
            
            output = np.stack((spectrum[:,0], baseline_subtract), axis=1)
            
            return output, background_STD, baseline_poly_vals

    ######################################################################################
    
    ### FUNCTION : REMOVING COSMIC RAYS
    def remove_cosmics(spec):
        
        peak_threshold = 1500
        
        peaks, peak_info = find_peaks(spec[:,1], height=peak_threshold, width=(0, 10))
        
        #print(peaks)
        #plt.plot(spec[:,0], spec[:,1])
        #plt.plot(spec[peaks, 0], spec[peaks, 1], "x")
        #plt.show()
        
        # make this less shite
        for p in peaks:
            print(p)
            print('Cosmic removed at wavenumber={:.4f}'.format(spec[p][0]))
            spec[p-2:p+3, 1] = np.mean([spec[p-3, 1], spec[p+3, 1]])
        #print(spec.shape)
        #spec_cosmicsub = np.delete(spec, peaks, axis=0)
        #print(spec_cosmicsub.shape)
        
        return spec
    ######################################################################################
    
    ### FUNCTIONS : LORENTZ FITTING 
    def lorentz_guess(data_peak):
        ## Guess parameters of lorentzian from line data selection   
        
        peak_amp = np.nanmax(data_peak[:, 1])
        ## minimum flux value in line region1
        
        peak_mdpt = np.nanmean(data_peak[:,0])
        ## mid point of line region
        
        peak_fwhm = (data_peak[0, 0] - data_peak[-1, 0]) * 0.25
        ## FWHM ~ 1/4 of data width
    
        #print(peak_amp, peak_mdpt, peak_fwhm)
        return (peak_amp, peak_mdpt, peak_fwhm)
    ######################################################################################
    
    ### FUNCTIONS : LORENTZ FITTING
    def lorentz_data_model(peak_guess, peak_wavenums, peak_intens, peak_x0bounds=[0, np.inf], peak_FWHMbounds=[0, np.inf]):    
        ## [0, np.inf] means that is it is not explicitly given a range it will still run assuming 0 to infinity
        ## Extract peak parameters (e.g. amplitude, mean, FWHM) and use those to model a Lorentzian fit
        
        peak_init = models.Lorentz1D(amplitude=peak_guess[0], x_0=peak_guess[1], fwhm=peak_guess[2])
        
        peak_init.bounds['x_0'] = peak_x0bounds # min wavenumber, max wavenumber of x_0 parameter
        peak_init.bounds['fwhm'] = peak_FWHMbounds # min FWHM, max FWHM of fwhm parameter
        ## Bounds applies constraints (i.e. min/max) to the fit, therefore instead of 'cutting' the data
        ## and making it fit within that window, it allows it to 'see' all the data but make it fit the peak
        ## within a specific region specified by the bounds
        
        fit_peak = fitting.LevMarLSQFitter()
        peak_result = fit_peak(peak_init, peak_wavenums, peak_intens)
        #fit_peak = fitting.SLSQPLSQFitter()
        #peak_result = fit_peak(peak_init, peak_wavenums, peak_intens, verblevel=0)
        
        return (peak_result)
    ######################################################################################
    
    ### FUNCTIONS : LORENTZ FITTING
    def lorentz_data_fit(peak_results, wavenumbers): 
        ## Output a lorentzian peak fit to data 
        
        final_lorentz = peak_results(wavenumbers)# + baseline_poly_vals
        
        return final_lorentz        
    ######################################################################################
    
    ###################################################################################### 
    ### SECTION 1 - DATA IMPORT AND SPECTRA ISOLATION
    
    ## CHOOSE a txt file in the current directory
    #map_data = np.genfromtxt('10a_1s_50x_MAPCu.txt', skip_header=1)
    
    ## OR
    
    ## CHOOSE a txt file from any directory with a select file window!
    map_data = np.genfromtxt(filedialog.askopenfile("r"))
    
    ######################################################################################
    
    ## ASK the user for a DoF
    degoffit = int(input("Degree of Fit:"))
    
    ## OR
    
    ## pre-select a DoF
    #degoffit = 6
    ######################################################################################
    
    ## np.copy - copies all the x-y values in data to 'coords'
    coords = np.copy(map_data[:,0:2])
    #print(map_data.shape)
    
    ## np.unique - finds the unique elements in the array (i.e. unique x-y pairs)
    uniquecoords = np.unique(coords, axis=0)
    
    uniquecoord_squared = float(len(uniquecoords)**(0.5))
    
    unqX = np.unique(uniquecoords[:,0])
    unqY = np.unique(uniquecoords[:,1])
    
    
    #print(uniquecoords.shape) 
    print("Total number of unique coord combinations =", len(uniquecoords))
    print("Unique X coords =",len(unqX), "\nUnique Y coords =", len(unqY))
    print("Unique coords squared =",uniquecoord_squared)      
    
    ### CREATE EMPTY ARRAYS TO BE FILLED BY SPECIFIED DATA LATER
    ## np.empty - returns a new array of a given shape and type (i.e. by the num. of uniquecoords)
    IntRat_TwoD_G = np.empty((len(uniquecoords), 1))
    IntRat_D_G = np.empty((len(uniquecoords), 1))
    
    FWHM_TwoD = np.empty((len(uniquecoords), 1))
    FWHM_G = np.empty((len(uniquecoords), 1))
    FWHM_D = np.empty((len(uniquecoords), 1))
    
    Pos_TwoD = np.empty((len(uniquecoords), 1))
    Pos_G = np.empty((len(uniquecoords), 1))
    Pos_D = np.empty((len(uniquecoords), 1))
    
    ## The wavenumber of each spectra will be the same, 'wavnum' isolates the wavenumber
    ## np.split - splits an array into mulitple arrays (i.e. by the num. of uniquecoords)
    
    Si_Calib_offset = 520.5 - float(input("Silicon Calibration Wavenumber:"))
    
    wavnums = np.split(map_data[:,-2], len(uniquecoords))[0] 
    print(wavnums[0])
    wavnums += Si_Calib_offset
    print(wavnums[0])
    
    ## np.mean - computes arithmetic mean (of ALL the spectra)
    AvSpectrum = np.mean(np.split(map_data[:,-1], len(uniquecoords)), axis=0)
    #print(AvSpectrum)
    
    ## np.stack - joins a sequence of arrays along an axis (i.e. along their columns)
    AvSpectrum =np.stack((wavnums,AvSpectrum), axis=1)
    print(AvSpectrum.shape)
    
    ### FIGURE : Each individual spectrum in the map has been averaged to create a single 'average' spectra
    plt.figure()
    plt.plot(AvSpectrum[:, 0], AvSpectrum[:, 1])
    plt.gca().axvspan(D_RangeBG[0], D_RangeBG[1], color='0.7', alpha=0.5)
    plt.gca().axvspan(G_RangeBG[0], G_RangeBG[1], color='0.7', alpha=0.5)
    plt.gca().axvspan(GPrime_RangeBG[0], GPrime_RangeBG[1], color='0.7', alpha=0.5)
    plt.gca().axvspan(TwoD_RangeBG[0], TwoD_RangeBG[1], color='0.7', alpha=0.5)
    plt.title("Average spectra")
    plt.xlabel("Wavenumber (cm-1)")
    plt.ylabel("Intensity (Arb.)")
    plt.savefig("AVERAGE_Raman_Scan.png")
    plt.show()
    
    for i in range(len(uniquecoords)):
    ## Does a test run through a section of the data (e.g. from index x to y (x,y))
    #for i in range(170,180):  
    
        ## TRYING TO REMOVE KNOWN DODGY SPECTRA corresponding to index number 
        ## (OTHERWISE DON'T NEED THE IF/ ELSE)
        #if i in [24, 29, 1110]: 
        #    spectrum = isolate_xy_spectra(uniquecoords, map_data)
        #    spectrum =  np.full_like(spectrum, np.nan, dtype=np.double)
        #else: 

        ## OR run all spectra in map
        
        if True:
            spectrum_raw = isolate_xy_spectra(uniquecoords[i], map_data)
            spectrum_raw[:,0] = wavnums
            
            print('Index = {}'.format(i))
            print(np.array(uniquecoords[i,:2])) 
        ###################################################################################
        ### SECTION 2 - BACKGROUND FITTING AND COSMIC SUBTRACTION
        
            ## collect noise here
            spectrum_BGsub_cos, Noise_STD, baseline = remove_bg_from_raw(spectrum_raw) 
            spectrum_BGCsub = remove_cosmics(spectrum_BGsub_cos)
        ################################################################################### 
        ### SECTION 3 - FIT PEAKS
        
            ## Fit D Peak
            ## Get the subset of data around the peak position
            boolean_Dpeak = data_range(spectrum_BGCsub, D_RangePF)
            data_Dpeak = spectrum_BGCsub[boolean_Dpeak]
            
            # Guess peak parameters
            guess_Dpeak = lorentz_guess(data_Dpeak)
            ## Make a fitter and fit the peak
            # def lorentz_data_model(peak_guess, peak_wavenums, peak_intens, 
            #                        peak_x0bounds=[0, np.inf], peak_FWHMbounds=[0, np.inf]):
            params_Dpeak = lorentz_data_model(guess_Dpeak, data_Dpeak[:,0], data_Dpeak[:,1],
                                              [1300, 1400])
            
            ## Fit G Peak
            ## same process as above
            boolean_Gpeak = data_range(spectrum_BGCsub, G_RangePF)
            data_Gpeak = spectrum_BGCsub[boolean_Gpeak]
            
            guess_Gpeak = lorentz_guess(data_Gpeak)
            
            params_Gpeak = lorentz_data_model(guess_Gpeak, data_Gpeak[:,0], data_Gpeak[:,1],
                                              [1550, 1650])
            
            ## Fit TwoD Peak
            ## again same as above
            boolean_TwoDpeak = data_range(spectrum_BGCsub, TwoD_RangePF)
            data_TwoDpeak = spectrum_BGCsub[boolean_TwoDpeak]
            
            guess_TwoDpeak = lorentz_guess(data_TwoDpeak)
            
            params_TwoDpeak = lorentz_data_model(guess_TwoDpeak, data_TwoDpeak[:,0], data_TwoDpeak[:,1],
                                                [2550, 2800]) 
                                                ## x_0 [2690, 2730] works
                
        ### QUALITY ASSURANCE - CHECK FIT QUALITY, MASK 'BAD' FITTING RESULTS AND ADD TO RESPECTIVE RESULT ARRAYS (See TOP) 
            ## X_0 values
            Pos_D[i] = params_Dpeak.x_0
            Pos_G[i] = params_Gpeak.x_0
            Pos_TwoD[i] = params_TwoDpeak.x_0
                   
            ## FWHM - check if above a threshold value and discard if so
            if params_Dpeak.fwhm > 80:
                FWHM_D[i] = np.nan
            else:
                FWHM_D[i] = params_Dpeak.fwhm
            
            if params_Gpeak.fwhm > 80:
                FWHM_G[i] = np.nan
            else:
                FWHM_G[i] = params_Gpeak.fwhm
            
            if params_TwoDpeak.fwhm > 80:
                FWHM_TwoD[i] = np.nan
            else:
                FWHM_TwoD[i] = params_TwoDpeak.fwhm
            
            ## RATIOS - check is amplitude is above threshold and discard if so
            ## Decide on noise limit
            if (params_TwoDpeak.amplitude <= 0.5*Noise_STD) or (params_Gpeak.amplitude <= 0.5*Noise_STD):
                IntRat_TwoD_G[i] = np.nan
            else:
                IntRat_TwoD_G[i] = params_TwoDpeak.amplitude/params_Gpeak.amplitude
                
            if (params_Dpeak.amplitude <= 0.5*Noise_STD) or (params_Gpeak.amplitude <= 0.5*Noise_STD):
                IntRat_D_G[i] = np.nan
            else:
                IntRat_D_G[i] = params_Dpeak.amplitude/params_Gpeak.amplitude
        ###################################################################################   
        ### SECTION 4 - PLOTTING!
        
            # def lorentz_data_fit(peak_results, wavenumbers):
            D_peak_fit = lorentz_data_fit(params_Dpeak, data_Dpeak[:,0])
            G_peak_fit = lorentz_data_fit(params_Gpeak, data_Gpeak[:,0])
            TwoD_peak_fit = lorentz_data_fit(params_TwoDpeak, data_TwoDpeak[:,0])
            
            ### FIGURE : 
            plt.figure()
            plt.plot(spectrum_BGCsub[:,0], spectrum_BGCsub[:,1], label = "Data")
            plt.plot(data_Dpeak[:,0], D_peak_fit, label = "D")
            plt.plot(data_Gpeak[:,0], G_peak_fit, label = "G")
            plt.plot(data_TwoDpeak[:,0], TwoD_peak_fit, label = "2D")
            plt.legend()
            plt.title("Background removed spectra at xy")
            plt.xlabel("Wavenumber (cm-1)")
            plt.ylabel("Intensity (Arb.)")
            plt.show()
            
            print('Noise = {:.1f}'.format(Noise_STD))
            
            print('\t\tx_0\tfwhm\tamplitude')
            print('{}\t\t{:.1f}\t{:.1f}\t{:.1f}'.format('D', params_Dpeak.x_0.value, params_Dpeak.fwhm.value, params_Dpeak.amplitude.value))
            print('{}\t\t{:.1f}\t{:.1f}\t{:.1f}'.format('G', params_Gpeak.x_0.value, params_Gpeak.fwhm.value, params_Gpeak.amplitude.value))
            print('{}\t\t{:.1f}\t{:.1f}\t{:.1f}'.format('TwoD', params_TwoDpeak.x_0.value, params_TwoDpeak.fwhm.value, params_TwoDpeak.amplitude.value))
            
            print("\n This is the 2D/G : {:.2f}".format(IntRat_TwoD_G[i][0]))
            print("\n This is the D/G : {:.2f}".format(IntRat_D_G[i][0]))
            print('##################################################')
            
    #print("\nThis is the IntRat_TwoD_G.shape:", IntRat_TwoD_G.shape)
    #print("This is the IntRat_D_G.shape:", IntRat_D_G.shape)
    #print("This is thw 2D FWHM", fwhmTwoD)
    
    #uniquecoords_TwoD_G = np.append(uniquecoords, IntRat_TwoD_G, axis=1)
    #uniquecoords_D_G = np.append(uniquecoords, IntRat_D_G, axis=1)
    #uniquecoords_TwoD_fwhm = np.append(uniquecoords, fwhmTwoD, axis=1)
    ######################################################################################
    
    MAP_Av_FWHM_TwoD= np.nanmean(FWHM_TwoD)
    MAP_Av_STD_FWHM_TwoD= np.nanstd(FWHM_TwoD)
    #print("MAP Average FWHW 2D:", MAP_Av_FWHM_TwoD, "error (STD)=", MAP_Av_STD_FWHM_TwoD)
    
    MAP_Av_pos_TwoD= np.nanmean(Pos_TwoD)
    MAP_Av_STD_pos_TwoD= np.nanstd(Pos_TwoD)
    #print("MAP Average pos 2D:", MAP_Av_pos_TwoD, "error (STD)=", MAP_Av_STD_pos_TwoD)
    ######################################################################################
    
    MAP_Av_FWHM_G= np.nanmean(FWHM_G)
    MAP_Av_STD_FWHM_G= np.nanstd(FWHM_G)
    #print("MAP Average FWHW G:", MAP_Av_FWHM_G, "error (STD)=", MAP_Av_STD_FWHM_G)
    
    MAP_Av_pos_G= np.nanmean(Pos_G)
    MAP_Av_STD_pos_G= np.nanstd(Pos_G)
    #print("MAP Average pos G:", MAP_Av_pos_G, "error (STD)=", MAP_Av_STD_pos_G)
    ######################################################################################
    
    MAP_Av_FWHM_D= np.nanmean(FWHM_D)
    MAP_Av_STD_FWHM_D= np.nanstd(FWHM_D)
    #print("MAP Average FWHW 2D:", MAP_Av_FWHM_D, "error (STD)=", MAP_Av_STD_FWHM_D)
    
    MAP_Av_pos_D= np.nanmean(Pos_D)
    MAP_Av_STD_pos_D= np.nanstd(Pos_D)
    #print("MAP Average pos D:", MAP_Av_pos_D, "error (STD)=", MAP_Av_STD_pos_D)
    ######################################################################################
    
    MAP_Av_IntRat_TwoD_G=np.nanmean(IntRat_TwoD_G)
    MAP_Av_STD_IntRat_TwoD_G=np.nanstd(IntRat_TwoD_G)
    #print("MAP Average Intensity 2D:G", MAP_Av_IntRat_TwoD_G, "error (STD)=", MAP_Av_STD_IntRat_TwoD_G)
    
    MAP_Av_IntRat_D_G=np.nanmean(IntRat_D_G)
    MAP_Av_STD_IntRat_D_G=np.nanstd(IntRat_D_G)
    #print("MAP Average Intensity D:G", MAP_Av_IntRat_D_G, "error (STD)=", MAP_Av_STD_IntRat_D_G)
    
    ######################################################################################
    
    ## Creates a x by y grid of the coordinates grid (DOES NOT ASSUME SQUARE)
    meshX, meshY = np.meshgrid(unqX,unqY)
    
    IntRat_2DG_copy = np.copy(IntRat_TwoD_G)
    #IntRat_2DG_copy[np.argwhere(IntRat_2DG_copy <= 0)] = 0.0000001
    IntRat_2DG_copy.shape=(int(len(unqY)), int(len(unqX)))
    
    IntRat_DG_copy = np.copy(IntRat_D_G)
    #IntRat_DG_copy[np.argwhere(IntRat_DG_copy <= 0)] = 0.0000001
    IntRat_DG_copy.shape=(int(len(unqY)), int(len(unqX)))
    
    std2D = np.copy(FWHM_TwoD)
    std2D.shape = (int(len(unqY)), int(len(unqX)))
    
    TwoD_pos = np.copy(Pos_TwoD)
    TwoD_pos.shape = (int(len(unqY)), int(len(unqX)))
    
    stdG = np.copy(FWHM_G)
    stdG.shape = (int(len(unqY)), int(len(unqX)))
    
    G_pos = np.copy(Pos_G)
    G_pos.shape = (int(len(unqY)), int(len(unqX)))
    
    stdD = np.copy(FWHM_D)
    stdD.shape = (int(len(unqY)), int(len(unqX)))
    
    D_pos = np.copy(Pos_D)
    D_pos.shape = (int(len(unqY)), int(len(unqX)))    
    ######################################################################################
    ## FIGURES (2D)
    
    fwhm_levels_2D=np.arange(35, 55, 0.5)
    
    ### FIGURE : Histogram (2D FWHM)
    plt.figure()
    plt.hist(std2D.flatten(), bins = fwhm_levels_2D, density=True)
    plt.title("2D FWHM")
    plt.savefig("Hist_2D_FWHM.png")
    plt.show()
    
    ### FIGURE :  Contour Plots (2D FWHM)
    plt.figure()
    CS_2Dfwhm = plt.contourf(unqX, unqY, std2D, levels=fwhm_levels_2D)
    ## make a colorbar for the contour lines
    CB_2Dfwhm = plt.colorbar(CS_2Dfwhm, shrink=1.0, extend='both')
    plt.title("2D FWHM")
    plt.savefig("Colour_2D_FWHM.png")
    plt.xlabel("x-coord")
    plt.ylabel("y-coord")
    plt.show()
    
    print("MAP Average FWHW 2D:", MAP_Av_FWHM_TwoD, "error (STD)=", MAP_Av_STD_FWHM_TwoD)
    np.savetxt("2D_FWHM_Hist.csv", std2D, delimiter=",")
    
    Position_2D=np.arange(2670, 2710, 0.5)
    
    ### FIGURE :  Histogram (2D Position)
    plt.figure()
    plt.hist(TwoD_pos.flatten(), bins = Position_2D, density=True)
    plt.title("2D Position")
    plt.savefig("Hist_2D_Position.png")
    plt.show()
    
    ### FIGURE :  Contour Plots (2D Position)
    plt.figure()
    CS_2D_pos = plt.contourf(unqX, unqY, TwoD_pos, levels=Position_2D)
    ## make a colorbar for the contour lines
    CB_2D_pos = plt.colorbar(CS_2D_pos, shrink=1.0, extend='both')
    plt.title("2D Position")
    plt.savefig("Colour_2D_pos.png")
    plt.xlabel("x-coord")
    plt.ylabel("y-coord")
    plt.show()
    
    print("MAP Average 2D Position:", MAP_Av_pos_TwoD, "error (STD)=", MAP_Av_STD_pos_TwoD)
    np.savetxt("2D_Position_Hist.csv", TwoD_pos, delimiter=",")   
    ######################################################################################
    ## FIGURES (G)
    
    fwhm_levels_G=np.arange(12,30, 0.5)
    
    ### FIGURE :  Histogram (G FWHM)
    plt.figure()
    plt.hist(stdG.flatten(), bins = fwhm_levels_G, density=True)
    plt.title("G FWHM")
    plt.savefig("Hist_G_FWHM.png")
    plt.show()
    
    ### FIGURE :  Contour Plots (G FWHM)
    plt.figure()
    CS_Gfwhm = plt.contourf(unqX, unqY, stdG, levels=fwhm_levels_G)
    ## make a colorbar for the contour lines
    CB_Gfwhm = plt.colorbar(CS_Gfwhm, shrink=1.0, extend='both')
    plt.title("G FWHM")
    plt.savefig("Colour_G_FWHM.png")
    plt.xlabel("x-coord")
    plt.ylabel("y-coord")
    plt.show()
    
    print("MAP Average FWHW G:", MAP_Av_FWHM_G, "error (STD)=", MAP_Av_STD_FWHM_G)
    np.savetxt("G_FWHM_Hist.csv", stdG, delimiter=",")
    
    Position_G=np.arange(1580, 1595, 0.5)
    
    ### FIGURE :  Histogram (G Position)
    plt.figure()
    plt.hist(G_pos.flatten(), bins = Position_G, density=True)
    plt.title("G Position")
    plt.savefig("Hist_G_Position.png")
    plt.show()
    
    ### FIGURE :  Contour Plots (G Position)
    plt.figure()
    CS_G_pos = plt.contourf(unqX, unqY, G_pos, levels=Position_G)
    ## make a colorbar for the contour lines
    CB_G_pos = plt.colorbar(CS_G_pos, shrink=1.0, extend='both')
    plt.title("G Position")
    plt.savefig("Colour_G_pos.png")
    plt.xlabel("x-coord")
    plt.ylabel("y-coord")
    plt.show()
    
    print("MAP Average G Position:", MAP_Av_pos_G, "error (STD)=", MAP_Av_STD_pos_G)
    np.savetxt("G_Position_Hist.csv", G_pos, delimiter=",")
    ######################################################################################
    ## FIGURES (D)
    
    fwhm_levels_D=np.arange(0, 80, 1)
    
    ### FIGURE :  Histogram (D FWHM)
    plt.figure()
    plt.hist(stdD.flatten(), bins = fwhm_levels_D, density=True)
    plt.title("D FWHM")
    plt.savefig("Hist_D_FWHM.png")
    plt.show()
    
    ### FIGURE :  Contour Plots (D FWHM)
    plt.figure()
    CS_Dfwhm = plt.contourf(unqX, unqY, stdD, levels=fwhm_levels_D)
    ## make a colorbar for the contour lines
    CB_Dfwhm = plt.colorbar(CS_Dfwhm, shrink=1.0, extend='both')
    plt.title("D FWHM")
    plt.savefig("Colour_D_FWHM.png")
    plt.xlabel("x-coord")
    plt.ylabel("y-coord")
    plt.show()
    
    print("MAP Average FWHW D:", MAP_Av_FWHM_D, "error (STD)=", MAP_Av_STD_FWHM_D)
    np.savetxt("D_FWHM_Hist.csv", stdD, delimiter=",")
    
    Position_D=np.arange(1330, 1370, 0.5)
    
    ### FIGURE :  Histogram (D Position)
    plt.figure()
    plt.hist(D_pos.flatten(), bins = Position_D, density=True)
    plt.title("D Position")
    plt.savefig("Hist_D_Position.png")
    plt.show()
    
    ### FIGURE :  Contour Plots (D Position)
    plt.figure()
    CS_D_pos = plt.contourf(unqX, unqY, D_pos, levels=Position_D)
    ## make a colorbar for the contour lines
    CB_D_pos = plt.colorbar(CS_D_pos, shrink=1.0, extend='both')
    plt.title("D Position")
    plt.savefig("Colour_D_pos.png")
    plt.xlabel("x-coord")
    plt.ylabel("y-coord")
    plt.show()
    
    print("MAP Average D Position:", MAP_Av_pos_D, "error (STD)=", MAP_Av_STD_pos_D)
    np.savetxt("D_Position_Hist.csv", D_pos, delimiter=",")
    ######################################################################################
    ## FIGURES (2D:G Ratio)
    
    TwoDG_ratio_levels=np.arange(0.6, 4, 0.05)
    
    ### FIGURE :  Histogram (2D:G)
    plt.figure()
    plt.hist(IntRat_2DG_copy.flatten(), bins = TwoDG_ratio_levels, density=True)
    plt.title("2D:G Ratio")
    plt.savefig("Hist_2D_G_Ratio.png")
    plt.show()
    
    ### FIGURE :  Contour Plots (2D:G)
    #level_values = [0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.8, 2.0, 2.5, 3, 3.5, 4, 5, 8, 40]
    #level_colours = ['k', 'tan', 'm', 'c', 'g', 'y', 'plum', 'navy', 'azure', 'khaki', 'red', 'darkcyan', 'violet', 'lavender', 'peru', 'blue', 'lawngreeen']
    plt.figure()
    CS_2DG = plt.contourf(unqX, unqY, IntRat_2DG_copy, levels=TwoDG_ratio_levels)
    ## make a colorbar for the contour lines
    CB_2DG = plt.colorbar(CS_2DG, shrink=1.0, extend='both')
    plt.title("2D:G Ratio")
    plt.savefig("ColourMap_2D_G_Ratio.png")
    plt.xlabel("x-coord")
    plt.ylabel("y-coord")
    plt.show()
    
    print("MAP Average Intensity 2D:G", MAP_Av_IntRat_TwoD_G, "error (STD)=", MAP_Av_STD_IntRat_TwoD_G)
    np.savetxt("2D_G_Ratio_Hist.csv", IntRat_2DG_copy, delimiter=",")   
    ######################################################################################
    ## FIGURES (D:G Ratio)
    
    DG_ratio_levels=np.arange(0, 0.75, 0.05)
    
    ### FIGURE :  Histogram (D:G)
    plt.figure()
    plt.hist(IntRat_DG_copy.flatten(), bins = DG_ratio_levels, density=True)
    plt.title("D:G Ratio")
    plt.savefig("Hist_D_G_Ratio.png")
    plt.show()
    
    ### FIGURE :  Contour Plots (2D:G)
    #level_values = [0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.8, 2.0, 2.5, 3, 3.5, 4, 5, 8, 40]
    #level_colours = ['k', 'tan', 'm', 'c', 'g', 'y', 'plum', 'navy', 'azure', 'khaki', 'red', 'darkcyan', 'violet', 'lavender', 'peru', 'blue', 'lawngreeen']
    plt.figure()
    CS_2DG = plt.contourf(unqX, unqY, IntRat_DG_copy, levels=DG_ratio_levels)
    ## make a colorbar for the contour lines
    CB_2DG = plt.colorbar(CS_2DG, shrink=1.0, extend='both')
    plt.title("D:G Ratio")
    plt.savefig("ColourMap_D_G_Ratio.png")
    plt.xlabel("x-coord")
    plt.ylabel("y-coord")
    plt.show()
    
    print("MAP Average Intensity D:G", MAP_Av_IntRat_D_G, "error (STD)=", MAP_Av_STD_IntRat_D_G)
    np.savetxt("D_G_Ratio_Hist.csv", IntRat_DG_copy, delimiter=",")
    
    ######################################################################################
    
    ######################################################################################
    ### FITTING THE SINGLE AVERAGED DATA
    
    ## RAMAN GRAPHENE (Peak (LORENTZ) Fit - slightly wider)
    AverageData_D_RangePF = [1250, 1450] 
    AverageData_G_RangePF = [1450, 1750]
    AverageData_GPrime_RangePF = [2300,2650]
    AverageData_TwoD_RangePF =[2600, 2800]
    
    ######################################################################################
    ### SECTION 2 (SETTING RANGES IN RAMAN PEAKS FOR THE FITTING! (SLIGHTLY WIDER THAN BG))
    
    Avspectrum_BGsub_cos, AvNoise_STD, Avbaseline_poly = remove_bg_from_raw(AvSpectrum) # collect noise here
    Avspectrum_BGCsub = remove_cosmics(Avspectrum_BGsub_cos)
    ######################################################################################
    ### SECTION 3 - FIT PEAKS
    
        # def lorentz_data_model(peak_guess, peak_wavenums, peak_intens, 
                            # peak_x0bounds=[0, np.inf], peak_FWHMbounds=[0, np.inf]):
    ## Fit D Peak
    ## Get the subset of data around the peak position
    Avboolean_Dpeak = data_range(Avspectrum_BGsub_cos, AverageData_D_RangePF)
    Avdata_Dpeak = spectrum_BGCsub[Avboolean_Dpeak]
            
    Avguess_Dpeak = lorentz_guess(Avdata_Dpeak)
    
    Avparams_Dpeak = lorentz_data_model(Avguess_Dpeak, Avdata_Dpeak[:,0], Avdata_Dpeak[:,1],
                                              [1300, 1400])
    ## Fit G Peak
    ## same process as above
    Avboolean_Gpeak = data_range(Avspectrum_BGsub_cos, AverageData_G_RangePF)
    Avdata_Gpeak = spectrum_BGCsub[Avboolean_Gpeak]
            
    Avguess_Gpeak = lorentz_guess(Avdata_Gpeak)
            
    Avparams_Gpeak = lorentz_data_model(Avguess_Gpeak, Avdata_Gpeak[:,0], Avdata_Gpeak[:,1],
                                              [1550, 1650])    
    ## Fit TwoD Peak
    ## again same as above
    Avboolean_TwoDpeak = data_range(Avspectrum_BGsub_cos, AverageData_TwoD_RangePF)
    Avdata_TwoDpeak = spectrum_BGCsub[Avboolean_TwoDpeak]
            
    Avguess_TwoDpeak = lorentz_guess(Avdata_TwoDpeak)
            
    Avparams_TwoDpeak = lorentz_data_model(Avguess_TwoDpeak, Avdata_TwoDpeak[:,0], Avdata_TwoDpeak[:,1],
                                                [2650, 2730]) 
                                                ## x_0 [2690, 2730] works
            
    ### QUALITY ASSURANCE - CHECK FIT QUALITY, MASK 'BAD' FITTING RESULTS AND ADD TO RESPECTIVE RESULT ARRAYS (See TOP) 
    ## X_0 values
    AvPos_D = Avparams_Dpeak.x_0
    AvPos_G = Avparams_Gpeak.x_0
    AvPos_TwoD = Avparams_TwoDpeak.x_0
                   
    ## FWHM - check if FWHM is above a threshold value and discard if not
    if Avparams_Dpeak.fwhm > 80:
                AvFWHM_D = np.nan
    else:
        AvFWHM_D = Avparams_Dpeak.fwhm
            
    if Avparams_Gpeak.fwhm > 80:
        AvFWHM_G = np.nan
    else:
        AvFWHM_G = Avparams_Gpeak.fwhm
            
    if Avparams_TwoDpeak.fwhm > 80:
        AvFWHM_TwoD = np.nan
    else:
        AvFWHM_TwoD = Avparams_TwoDpeak.fwhm
            
    ## RATIOS - check if amplitude is above threshold and discard if not
    ## Decide on noise limit
    if (Avparams_TwoDpeak.amplitude <= 0.5*AvNoise_STD) or (Avparams_Gpeak.amplitude <= 0.5*AvNoise_STD):
        AvIntRat_TwoD_G = np.nan
    else:
        AvIntRat_TwoD_G = Avparams_TwoDpeak.amplitude/Avparams_Gpeak.amplitude
                
    if (Avparams_Dpeak.amplitude <= 0.5*AvNoise_STD) or (Avparams_Gpeak.amplitude <= 0.5*AvNoise_STD):
        AvIntRat_D_G = np.nan
    else:
        AvIntRat_D_G = Avparams_Dpeak.amplitude/Avparams_Gpeak.amplitude
    
    ### FIGURE : Averaged data (AvSpecrum)
    plt.figure()
    plt.plot(AvSpectrum[:, 0], AvSpectrum[:, 1])
    plt.plot(AvSpectrum[:, 0], Avbaseline_poly, label = "Poly Fit")
    plt.legend()
    plt.title("Average Spectra")
    plt.xlabel("Wavenumber (cm-1)")
    plt.ylabel("Intensity (Arb.)")
    plt.savefig("AVERAGE_Raman_Scan.png")
    plt.show()
    
    ## def lorentz_data_fit(peak_results, wavenumbers):
    AvD_peak_fit = lorentz_data_fit(Avparams_Dpeak, Avdata_Dpeak[:,0])
    AvG_peak_fit = lorentz_data_fit(Avparams_Gpeak, Avdata_Gpeak[:,0])
    AvTwoD_peak_fit = lorentz_data_fit(Avparams_TwoDpeak, Avdata_TwoDpeak[:,0])
    
    ### FIGURE : AvSpecrum Baseline subtracted data
    plt.figure()
    plt.plot(Avspectrum_BGCsub[:,0], Avspectrum_BGCsub[:,1])
    plt.title("Average Spectra - Background Subtracted ")
    plt.xlabel("Wavenumber (cm-1)")
    plt.ylabel("Intensity (Arb.)")
    plt.show()
    
    
    ### FIGURE : AvSpecrum Lorentzian fitted data
    plt.figure()
    plt.plot(Avspectrum_BGCsub[:,0], Avspectrum_BGCsub[:,1], label = "Data")
    plt.plot(Avdata_Dpeak[:,0], AvD_peak_fit, label = "D")
    plt.plot(Avdata_Gpeak[:,0], AvG_peak_fit, label = "G")
    plt.plot(Avdata_TwoDpeak[:,0], AvTwoD_peak_fit, label = "2D")
    plt.legend()
    plt.title("Average Spectra - Fitted")
    plt.xlabel("Wavenumber (cm-1)")
    plt.ylabel("Intensity (Arb.)")
    plt.savefig("Average_BG_Subtract.png")
    plt.show()
    
    print("\n MAP Average FWHW 2D:", MAP_Av_FWHM_TwoD, "error (STD)=", MAP_Av_STD_FWHM_TwoD)
    print("Averaged scans FWHW 2D:", Avparams_TwoDpeak.fwhm)
    
    print("\n MAP Average 2D Position:", MAP_Av_pos_TwoD, "error (STD)=", MAP_Av_STD_pos_TwoD)
    print("Averaged scans 2D Position:", Avparams_TwoDpeak.x_0)
    print("_____________________________________________________________________________")
    ######################################################################################
    
    print("\n MAP Average FWHW G:", MAP_Av_FWHM_G, "error (STD)=", MAP_Av_STD_FWHM_G)
    print("Averaged scans FWHW G:", Avparams_Gpeak.fwhm)
    
    print("\n MAP Average G Position:", MAP_Av_pos_G, "error (STD)=", MAP_Av_STD_pos_G)
    print("Averaged scans G Position:", Avparams_Gpeak.x_0)
    print("_____________________________________________________________________________")
    ######################################################################################
    
    print("\n MAP Average FWHW D:", MAP_Av_FWHM_D, "error (STD)=", MAP_Av_STD_FWHM_D)
    print("Averaged scans FWHW D:", Avparams_Dpeak.fwhm)
    
    print("\n MAP Average D Position:", MAP_Av_pos_D, "error (STD)=", MAP_Av_STD_pos_D)
    print("Averaged scans D Position:", Avparams_Dpeak.x_0)
    print("_____________________________________________________________________________")
    ######################################################################################
    
    print("\n MAP Intensity D:G", MAP_Av_IntRat_D_G, "error (STD)=", MAP_Av_STD_IntRat_D_G)
    print("Averaged scans Intensity D:G", AvIntRat_D_G)
    
    print("\n MAP Intensity 2D:G", MAP_Av_IntRat_TwoD_G, "error (STD)=", MAP_Av_STD_IntRat_TwoD_G)
    print("Averaged scans Intensity 2D:G", AvIntRat_TwoD_G)    
    ######################################################################################
    
    ######################################################################################
    ### EXPORT
    
    print(Avdata_Dpeak[:,0].shape)
    print(AvD_peak_fit.shape)
    
    print(wavnums.shape)
    print(Avspectrum_BGCsub.shape)
    print(Avbaseline_poly.shape)
    
    #Splice wavenumbers onto Data
    AvSpec_polyfit=np.stack((AvSpectrum[:,0], AvSpectrum[:,1], Avbaseline_poly), axis=1)
    AvSpec_baseline_subtract1=np.stack((Avspectrum_BGCsub[:,0], Avspectrum_BGCsub[:,1]), axis=1)
    AvSpec_D_peak_fit1=np.stack((Avdata_Dpeak[:,0], AvD_peak_fit), axis=1)
    AvSpec_G_peak_fit1=np.stack((Avdata_Gpeak[:,0], AvG_peak_fit), axis=1)
    AvSpec_TwoD_peak_fit1=np.stack((Avdata_TwoDpeak[:,0], AvTwoD_peak_fit), axis=1)
    
    np.savetxt("Average_Spectrum.csv", AvSpec_polyfit, delimiter=",")
    np.savetxt("BS_Average_Spectrum.csv", AvSpec_baseline_subtract1, delimiter=",")
    np.savetxt("BS_Average_SpectrumD.csv", AvSpec_D_peak_fit1, delimiter=",")
    np.savetxt("BS_Average_SpectrumG.csv", AvSpec_G_peak_fit1, delimiter=",")
    np.savetxt("BS_Average_Spectrum2D.csv", AvSpec_TwoD_peak_fit1, delimiter=",")
    
    import csv
    with open("MAP_Data.csv", "w", newline="") as csvfile:
        datawriter = csv.writer(csvfile, delimiter=",",
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        datawriter.writerow([ "MAP Average FWHW 2D:", MAP_Av_FWHM_TwoD, "error (STD)=", MAP_Av_STD_FWHM_TwoD])
        datawriter.writerow(["Averaged scans FWHW 2D:", Avparams_TwoDpeak.fwhm])
    
        datawriter.writerow(["\n MAP Average 2D Position:", MAP_Av_pos_TwoD, "error (STD)=", MAP_Av_STD_pos_TwoD])
        datawriter.writerow(["Averaged scans 2D Position:", Avparams_TwoDpeak.x_0])
    
        datawriter.writerow(["\n MAP Average FWHW G:", MAP_Av_FWHM_G, "error (STD)=", MAP_Av_STD_FWHM_G])
        datawriter.writerow(["Averaged scans FWHW G:", Avparams_Gpeak.fwhm])
    
        datawriter.writerow(["\n MAP Average G Position:", MAP_Av_pos_G, "error (STD)=", MAP_Av_STD_pos_G])
        datawriter.writerow(["Averaged scans G Position:", Avparams_Gpeak.x_0])
    
        datawriter.writerow(["\n MAP Average FWHW D:", MAP_Av_FWHM_D, "error (STD)=", MAP_Av_STD_FWHM_D])
        datawriter.writerow(["Averaged scans FWHW D:", Avparams_Dpeak.fwhm])
    
        datawriter.writerow(["\n MAP Average D Position:", MAP_Av_pos_D, "error (STD)=", MAP_Av_STD_pos_D])
        datawriter.writerow(["Averaged scans D Position:", Avparams_Dpeak.x_0])
    
        datawriter.writerow(["\n MAP Intensity 2D:G", MAP_Av_IntRat_D_G, "error (STD)=", MAP_Av_STD_IntRat_D_G])
        datawriter.writerow(["Averaged scans Intensity 2D:G", AvIntRat_D_G])
    
        datawriter.writerow(["\n MAP Intensity 2D:G", MAP_Av_IntRat_TwoD_G, "error (STD)=", MAP_Av_STD_IntRat_TwoD_G])
        datawriter.writerow(["Averaged scans Intensity 2D:G", AvIntRat_TwoD_G]) 
