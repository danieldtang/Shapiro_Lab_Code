close all;
clear all;
%IMPLEMENTATION RFOVE method [1]
% [1] C. Panagiotakis and A.A. Argyros, "Region-based Fitting of Overlapping Ellipses and its
% Application to Cells Segmentation", Image and Vision Computing, Elsevier, vol. 93, pp. 103810, 2020.
%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Method Selection
%Set METHOD = 0 to only test the Segmentation Stage,
%Set METHOD = 1 to run DEFA method,
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

METHOD = 1;
METHODSEG = 4;
global Constraints
Constraints = [250 0.1 0.2];
%Constraints(1) = areaLim ( < 250)
%Constraints(2) = min area / max area ratio e.g. < 0.1
%Constraints(3) = max overlaping > 0.2

AICBIC_SELECTION = 1; %Set AICBIC_SELECTION = 1, to use AIC is selected else BIC is used

set(0,'DefaultFigureColormap',jet);

DataDir = ;
ResultsDir = DataDir;

NeighborhoodSize = 151;

[I] = imread(sprintf('%s%s.png',DataDir,fname));
GT = imread(sprintf('%s%s.png',DataDir,fnameGT));
[ GT ] = correctGT( GT);

[IClustTotal,totEll,INITSEG] = runMainAlgo(imgaussfilt(I,2),AICBIC_SELECTION,METHOD,METHODSEG,NeighborhoodSize,0.5,0);
[REC, PR, F1, Overlap,BP,BDE,RECL,PRL,F1L,LGT] = getInitSegmentationStats(GT,INITSEG,IClustTotal);
[Jaccard, MAD, Hausdorff, DiceFP,DiceFN,FP,FN,LGT] =getStats(GT,INITSEG,IClustTotal);
%myImWrite(I,IClustTotal,GT,ResultsDir,fname,0);
myImWriteOnRealImages(I,IClustTotal,LGT,ResultsDir,fname,0 );
%myImWriteOnRealImagesSynthEL(I,totEll,pgt,ResultsDir,fname,1 );

close all;
statsE{id} = totEll;
Fnames{id} = fname;
statsPan(id,1:9) = [REC, PR, F1, Overlap,BP,BDE,RECL,PRL,F1L];
stats(id,1:7) = [Jaccard, MAD, Hausdorff, DiceFP,DiceFN,FP,FN];
id = id+1;
save(sprintf('%sAIC.mat', ResultsDir),'stats','statsE','statsPan','Fnames');

