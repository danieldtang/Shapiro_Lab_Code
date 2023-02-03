clear, close all, clc;

addpath 'C:\Users\mgs-lab-admin\Desktop\Daniel_Rotation\Code\CaImAn-MATLAB'
addpath 'C:\Users\mgs-lab-admin\Desktop\Daniel_Rotation\Code\RFOVE_Segmentation'

file_dir = "/Users/mgs-lab-admin/Desktop/Daniel_Rotation/Data/Daniel_Hao/US_agonist/";
% filename = "hsTRPA1_1_stim.tif";          % insert path to tiff stack here
% % rfp_filepath = nam(1:end-8)+"red.tif";
% rfp_filepath = "hsTRPA1_1_red.tif";
gfp_filepath = "US_agonist_correlation.mat";
sframe=1;						% user input: first frame to read (optional, default 1)
num2read=2000;	



% warning('off','all')
% image_gfp = read_file(file_dir + gfp_filepath,sframe,num2read);
% warning('on','all')
% 
% c = 250;
% kernel = ones(c)/(c^2);
% % Filter the image.  Need to cast to single so it can be floating point
% % which allows the image to have negative values.
% blurred_image_gfp = imfilter(single(image_gfp), kernel,"replicate" );
% hp_image_gfp = single(image_gfp) - blurred_image_gfp;
data = load(file_dir + gfp_filepath);
image_gfp = data.Cn;

% zscore_image_gfp = zscore(double(hp_image_gfp));


%%

%IMPLEMENTATION RFOVE method [1]
% [1] C. Panagiotakis and A.A. Argyros, "Region-based Fitting of Overlapping Ellipses and its
% Application to Cells Segmentation", Image and Vision Computing, Elsevier, vol. 93, pp. 103810, 2020.
%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Method Selection
%Set METHOD = 0 to only test the Segmentation Stage,
%Set METHOD = 1 to run DEFAmethod,
%Set METHOD = 2 to run AEFA method,
%else you run EMAR method
%Set METHODSEG = 0  to run ICPR 2010 method
%Set METHODSEG = 1  to run OTSU method
%Set METHODSEG = 2 to run Adaptive Thresh method, [2]
%Set METHODSEG = 3 to run Adaptive Thresh+extra method LADA+ [2],
%Set METHODSEG = 4 to run ICIP 2018 method [2],
%[2] C. Panagiotakis and A. Argyros, Cell Segmentation via Region-based Ellipse Fitting, IEEE International Conference on Image Processing, 2018.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% SET PARAMETERS (selected for Dataset NIH3T3)

METHOD = 0;
METHODSEG = 4;
global Constraints
Constraints = [250 0.1 0.2];
%Constraints(1) = areaLim ( < 250)
%Constraints(2) = min area / max area ratio e.g. < 0.1
%Constraints(3) = max overlaping > 0.2

AICBIC_SELECTION = 1; %Set AICBIC_SELECTION = 1, to use AIC is selected else BIC is used

set(0,'DefaultFigureColormap',jet);

NeighborhoodSize = 151;

% [I] = imread(sprintf('%s%s.png',DataDir,fname));

[IClustTotal,totEll,INITSEG] = runMainAlgo(imgaussfilt(image_gfp,2),AICBIC_SELECTION,METHOD,METHODSEG,NeighborhoodSize,0.5,0);
% [REC, PR, F1, Overlap,BP,BDE,RECL,PRL,F1L,LGT] = getInitSegmentationStats(GT,INITSEG,IClustTotal);
% [Jaccard, MAD, Hausdorff, DiceFP,DiceFN,FP,FN,LGT] =getStats(GT,INITSEG,IClustTotal);

imshow(IClustTotal)