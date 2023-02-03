% complete pipeline for calcium imaging data pre-processing
clc, close all, clear all;

tic

addpath(genpath('../NoRMCorre'));               % add the NoRMCorre motion correction package to MATLAB path
addpath(genpath('utilities'));
addpath(genpath('deconvolution'));
gcp;                                            % start a parallel engine

dev_mode = true;

file_dir = "/Users/mgs-lab-admin/Desktop/Daniel_Rotation/Data/Daniel_Hao/US_agonist/";
filename = "hsTRPA1_pre500uLGel_AITC11uM_3_agns";          % insert path to tiff stack here
file_type = ".tif";
% rfp_filepath = nam(1:end-8)+"red.tif";
rfp_filepath = "hsTRPA1_pre500uLGel_AITC11uM_3_red";
gfp_filepath = "hsTRPA1_pre500uLGel_AITC11uM_3_grn";

sframe=1;						% user input: first frame to read (optional, default 1)
num2read=2000;					% user input: how many frames to read   (optional, default until the end)
numFiles = 1;

warning('off','all')

Y = read_file(file_dir+filename+file_type,sframe,num2read);
image_rfp = read_file(file_dir + rfp_filepath+file_type,sframe,num2read);
% image_gfp = read_file(file_dir + gfp_filepath,sframe,num2read);

warning('on','all')

% if dev_mode
%     Y = Y(1:550,1:650,:);
% end

FOV = size(Y);
[d1,d2,T] = size(Y);                                % dimensions of dataset
d = d1*d2;                                          % total number of pixels

%% motion correct (and save registered h5 files as 2d matrices (to be used in the end)..)
% register files one by one. use template obtained from file n to
% initialize template of file n + 1; 

motion_correct = true;                            % perform motion correction
non_rigid = true;                                 % flag for non-rigid motion correction
output_type = 'h5';                               % format to save registered files

if non_rigid; append = '_nr'; else; append = '_rig'; end        % use this to save motion corrected files

options_mc = NoRMCorreSetParms('d1',d1,'d2',d2,'grid_size',[128,128],'init_batch',200,...
                'overlap_pre',32,'mot_uf',4,'bin_width',200,'max_shift',24,'max_dev',8,'us_fac',50,...
                'output_type',output_type);

template = [];
col_shift = [];

output_filename = char(file_dir+"motion_corrected/"+filename+'.'+output_type);
options_mc = NoRMCorreSetParms(options_mc,'output_filename',output_filename,'h5_filename','','tiff_filename',''); % update output file name
if motion_correct
    [M,shifts,template,options_mc,col_shift] = normcorre_batch_even(char(file_dir+filename+file_type),options_mc,template);
    save(file_dir+"motion_corrected/"+filename+'_shifts'+'.mat','shifts','-v7.3');           % save shifts of each file at the respective folder
else    % if files are already motion corrected convert them to h5
    convert_file(char(file_dir+filename+file_type),'h5',file_dir+"motion_corrected/"+filename+'_mc.h5');
end


%% downsample h5 files and save into a single memory mapped matlab file

if motion_correct
    registered_files = subdir(char(file_dir+"motion_corrected/"+"*"+"."+output_type));  % list of registered files (modify this to list all the motion corrected files you need to process)
else
    registered_files = subdir(char(file_dir+"motion_corrected/",'*_mc.h5'));
end
    
fr = 5;                                         % frame rate
tsub = 1;                                        % degree of downsampling (for 30Hz imaging rate you can try also larger, e.g. 8-10)

data = read_file(registered_files(1).name);
info = h5info(registered_files(1).name);
dims = info.Datasets.Dataspace.Size;
ndimsY = length(dims);
Ts = dims(end);

% ds_filename = char(file_dir+'/ds_data.mat');
% data_type = class(read_file(registered_files(1).name,1,1));
% data = matfile(ds_filename,'Writable',true);
% data.Y  = zeros(FOV,data_type);
% data.Yr = zeros(prod(FOV),data_type);
% data.sizY = FOV;
% F_dark = Inf;                                    % dark fluorescence (min of all data)
% batch_size = 2000;                               % read chunks of that size
% batch_size = round(batch_size/tsub)*tsub;        % make sure batch_size is divisble by tsub
% Ts = zeros(numFiles,1);                          % store length of each file
% cnt = 0;                                         % number of frames processed so far
% tt1 = tic;
% 
% for i = 1:numFiles
%     name = registered_files(i).name;
%     info = h5info(name);
%     dims = info.Datasets.Dataspace.Size;
%     ndimsY = length(dims);                       % number of dimensions (data array might be already reshaped)
%     Ts(i) = dims(end);
% %     Ysub = zeros(FOV(1),FOV(2),floor(Ts(i)/tsub),data_type);
% %     data.Y(FOV(1),FOV(2),sum(floor(Ts/tsub))) = zeros(1,data_type);
% %     data.Yr(prod(FOV),sum(floor(Ts/tsub))) = zeros(1,data_type);
%     cnt_sub = 0;
%     for t = 1:batch_size:Ts(i)
%         Y = read_file(name,t,min(batch_size,Ts(i)-t+1));    
%         F_dark = min(nanmin(Y(:)),F_dark);
% %         ln = size(Y,ndimsY);
% %         Y = reshape(Y,[FOV,ln]);
% %         Y = cast(downsample_data(Y,'time',tsub),data_type);
% %         ln = size(Y,3);
% %         Ysub(:,:,cnt_sub+1:cnt_sub+ln) = Y;
% %         cnt_sub = cnt_sub + ln;
%     end
% %     data.Y(:,:,cnt+1:cnt+cnt_sub) = Ysub;
% %     data.Yr(:,cnt+1:cnt+cnt_sub) = reshape(Ysub,[],cnt_sub);
%     toc(tt1);
% %     cnt = cnt + cnt_sub;
% %     data.sizY(1,3) = cnt;
% end
% data.F_dark = F_dark;
%% now run CNMF on patches on the downsampled file, set parameters first

sizY = FOV;                       % size of data matrix
patch_size = [550,650];                   % size of each patch along each dimension (optional, default: [32,32])
overlap = [10,10];                        % amount of overlap in each dimension (optional, default: [4,4])

patches = construct_patches(sizY(1:end-1),patch_size,overlap);
K = 500;                                        % number of components to be found
tau = 6;                                        % orig 8, std of gaussian kernel (half size of neuron) 
p = 2;                                          % order of autoregressive system (p = 0 no dynamics, p=1 just decay, p = 2, both rise and decay)
merge_thr = 0.97;                                % merging threshold
gnb = 3;                                        % number of global background components
min_SNR = 3; % originally 2
max_size_thr = 20; % originally 300
min_size_thr = 10;  % originally 10

options = CNMFSetParms(...
    'd1',sizY(1),'d2',sizY(2),...
    'deconv_method','constrained_foopsi',...    % neural activity deconvolution method
    'p',p,...                                   % order of calcium dynamics
    'ssub',2,...                                % spatial downsampling when processing
    'tsub',tsub,...                                % further temporal downsampling when processing
    'merge_thr',merge_thr,...                   % merging threshold
    'gSig',tau,... 
    'max_size_thr',max_size_thr,'min_size_thr',min_size_thr,...    % max/min acceptable size for each component
    'spatial_method','regularized',...          % method for updating spatial components
    'df_prctile',50,...                         % take the median of background fluorescence to compute baseline fluorescence 
    'fr',fr/tsub,...                            % downsamples
    'space_thresh',0.35,...                     % 0.35, space correlation acceptance threshold
    'min_SNR',min_SNR,...                           % trace SNR acceptance threshold
    'cnn_thr',0.2,...                           % cnn classifier acceptance threshold
    'nb',1,...                                  % number of background components per patch
    'gnb',gnb,...                               % number of global background components
    'decay_time',3 ...                         % 0.5, length of typical transient for the indicator used
    );

%% Run on patches (the main work is done here)

[A,b,C,f,S,P,RESULTS,YrA, F_dark] = run_CNMF_patches(data,K,patches,tau,0,options);  % do not perform deconvolution here since
                                                                               % we are operating on downsampled data
%% compute correlation image on a small sample of the data (optional - for visualization purposes) 
Cn = correlation_image_max(data,8);

%% classify components

rval_space = classify_comp_corr(data,A,C,b,f,options);
ind_corr = rval_space > options.space_thresh;           % components that pass the correlation test
                                                        % this test will keep processes
                                        
%% further classification with cnn_classifier
try  % matlab 2017b or later is needed
    [ind_cnn,value] = cnn_classifier(A,FOV,'cnn_model',options.cnn_thr);
catch
    ind_cnn = true(size(A,2),1);                        % components that pass the CNN classifier
end     
                            
%% event exceptionality

fitness = compute_event_exceptionality(C+YrA,options.N_samples_exc,options.robust_std);
ind_exc = (fitness < options.min_fitness);

%% select components

keep = (ind_corr | ind_cnn) & ind_exc;
%% run GUI for modifying component selection (optional, close twice to save values)
% run_GUI = false;
% if run_GUI
%     Coor = plot_contours(A,Cn,options,1); close;
%     GUIout = ROI_GUI(A,options,Cn,Coor,keep,ROIvars);   
%     options = GUIout{2};
%     keep = GUIout{3};    
% end

%% view contour plots of selected and rejected components (optional)
throw = ~keep;
throw = [];
Coor_k = [];
Coor_t = [];
figure;
    ax1 = subplot(121); plot_contours(A(:,keep),Cn,options,0,[],Coor_k,[],1,find(keep)); title('Selected components','fontweight','bold','fontsize',14);
    ax2 = subplot(122); plot_contours(A(:,throw),Cn,options,0,[],Coor_t,[],1,find(throw));title('Rejected components','fontweight','bold','fontsize',14);
    linkaxes([ax1,ax2],'xy')
    
%% keep only the active components    

% A_keep = A(:,keep);
% C_keep = C(keep,:);

A_keep = A;
C_keep = C;

%% extract residual signals for each trace

if exist('YrA','var') 
    R_keep = YrA(keep,:); 
else
    R_keep = compute_residuals(data,A_keep,b,C_keep,f);
end
    
%% extract fluorescence on native temporal resolution

options.fr = options.fr*tsub;                   % revert to origingal frame rate
N = size(C_keep,1);                             % total number of components
T = FOV(3);                                    % total number of timesteps
C_full = imresize(C_keep,[N,T]);                % upsample to original frame rate
R_full = imresize(R_keep,[N,T]);                % upsample to original frame rate
F_full = C_full + R_full;                       % full fluorescence
f_full = imresize(f,[size(f,1),T]);             % upsample temporal background

S_full = zeros(N,T);

P.p = 0;
ind_T = [0;cumsum(Ts(:))];
options.nb = options.gnb;
for i = 1:numFiles
    inds = ind_T(i)+1:ind_T(i+1);   % indeces of file i to be updated
    [C_full(:,inds),f_full(:,inds),~,~,R_full(:,inds)] = update_temporal_components_fast(registered_files(i).name,A_keep,b,C_full(:,inds),f_full(:,inds),P,options);
    disp(['Extracting raw fluorescence at native frame rate. File ',num2str(i),' out of ',num2str(numFiles),' finished processing.'])
end

%% extract DF/F and deconvolve DF/F traces

[F_dff,F0] = detrend_df_f(A_keep,[b,ones(prod(FOV(1:2)),1)],C_full,[f_full;-double(F_dark)*ones(1,T)],R_full,options);

C_dec = zeros(N,T);         % deconvolved DF/F traces
S_dec = zeros(N,T);         % deconvolved neural activity
bl = zeros(N,1);            % baseline for each trace (should be close to zero since traces are DF/F)
neuron_sn = zeros(N,1);     % noise level at each trace
g = cell(N,1);              % discrete time constants for each trace
if p == 1; model_ar = 'ar1'; elseif p == 2; model_ar = 'ar2'; else; error('This order of dynamics is not supported'); end

for i = 1:N
    spkmin = options.spk_SNR*GetSn(F_dff(i,:));
    lam = choose_lambda(exp(-1/(options.fr*options.decay_time)),GetSn(F_dff(i,:)),options.lam_pr);
    [cc,spk,opts_oasis] = deconvolveCa(F_dff(i,:),model_ar,'method','thresholded','optimize_pars',true,'maxIter',20,...
                                'window',150,'lambda',lam,'smin',spkmin);
    bl(i) = opts_oasis.b;
    C_dec(i,:) = cc(:)' + bl(i);
    S_dec(i,:) = spk(:);
    neuron_sn(i) = opts_oasis.sn;
    g{i} = opts_oasis.pars(:)';
    disp(['Performing deconvolution. Trace ',num2str(i),' out of ',num2str(N),' finished processing.'])
end

toc