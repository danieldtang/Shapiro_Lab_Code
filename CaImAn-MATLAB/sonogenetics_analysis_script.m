clear, close all, clc;
%% load file
gcp;                            % start cluster
addpath(genpath('utilities'));
addpath(genpath('deconvolution'));

nam = '/Users/mgs-lab-admin/Desktop/Daniel_Rotation/Data/Daniel_Hao/US_gel/hsTRPA1_1_stim.tif';          % insert path to tiff stack here
rfp_filepath = nam(1:end-8)+"red.tif";

devMode = true;

sframe=1;						% user input: first frame to read (optional, default 1)
num2read=2000;					% user input: how many frames to read   (optional, default until the end)
Y = read_file(nam,sframe,num2read);
image_rfp = read_file(rfp_filepath,sframe,num2read);

%% Preprocess image
image_rfp_scaled = image_rfp - mean(image_rfp, "all");

% figure;
% im = image(image_rfp_scaled, "CDataMapping","scaled");

% rfp_fft = fft2(image_rfp_scaled);
% % figure;
% % imagesc(abs(fftshift(rfp_fft)))
% rfp_fft(1,1) = 0;
% rec_image_rfp = ifft2(rfp_fft);

c = 7;
kernel = -ones(3)/(c+1);
kernel(2,2) = c/(c+1);
% Filter the image.  Need to cast to single so it can be floating point
% which allows the image to have negative values.
% filtered_image_rfp = imfilter(single(image_rfp_scaled), kernel);
% filtered_image_rfp = zscore(filtered_image_rfp);
% filtered_image_rfp = max(max(filtered_image_rfp)) - filtered_image_rfp;
zscore_image_rfp = zscore(double(image_rfp_scaled));

% figure;
% im2 = image(filtered_rfp_image, "CDataMapping","scaled");

%Y = Y - min(Y(:));
if ~isa(Y,'single');    Y = single(Y);  end         % convert to single

% if devMode
%     [d1,d2,~] = size(Y);
%     Y = Y(round(3*d1/8):round(5*d1/8),round(3*d2/8):round(5*d2/8),:);
%     image_rfp = image_rfp(round(3*d1/8):round(5*d1/8),round(3*d2/8):round(5*d2/8),:);
%     %     filtered_image_rfp = filtered_image_rfp(round(3*d1/8):round(5*d1/8),round(3*d2/8):round(5*d2/8),:);
%     zscore_image_rfp = zscore_image_rfp(round(3*d1/8):round(5*d1/8),round(3*d2/8):round(5*d2/8),:);
% end

figure;
im1 = image(image_rfp, "CDataMapping","scaled");
% figure;
% im2 = image(filtered_image_rfp, "CDataMapping","scaled");

[d1,d2,T] = size(Y);                                % dimensions of dataset
d = d1*d2;                                          % total number of pixels

%% Set parameters

K = 400; % optimized                              % number of components to be found
tau = 7;                                          % std of gaussian kernel (half size of neuron)
p = 2; % default is 2
min_SNR = 3; % optimized/default is 3
paramName = "tau_screen/";

rfp_threshold_1 = 0.5;                            % default (0.5) avg fluorescence value within identified ROIs
% rfp_threshold_2 = 1.1;                        % default (1-1.2) avg fluorescence value within identified ROIs
rfp_threshold_2 = 0.1;
rfp_thresholds = [rfp_threshold_1, rfp_threshold_2];

make_movie = false;

options = CNMFSetParms(...
    'd1',d1,'d2',d2,...                         % dimensionality of the FOV
    'p',p,...                                   % order of AR dynamics
    'gSig',tau,...                              % half size of neuron
    'merge_thr',0.80,...                        % merging threshold
    'nb',2,...                                  % number of background components
    'min_SNR',min_SNR,...                             % minimum SNR threshold
    'space_thresh',0.5,...                      % space correlation threshold
    'cnn_thr',0.2...                            % threshold for CNN classifier
    );
%% Data pre-processing

[P,Y] = preprocess_data(Y,p);
%% fast initialization of spatial components using greedyROI and HALS

[Ain,Cin,bin,fin,center] = initialize_components(Y,K,tau,options,P);  % initialize

% display centers of found components
Cn =  correlation_image(Y); %reshape(P.sn,d1,d2);  %max(Y,[],3); %std(Y,[],3); % image statistic (only for display purposes)
figure;imagesc(Cn);
axis equal; axis tight; hold all;
scatter(center(:,2),center(:,1),'mo');
title('Center of ROIs found from initialization algorithm');
drawnow;


%% manually refine components (optional)
refine_components = false;  % flag for manual refinement
if refine_components
    [Ain,Cin,center] = manually_refine_components(Y,Ain,Cin,center,Cn,tau,options);
end

%% update spatial components
Yr = reshape(Y,d,T);
[A,b,Cin] = update_spatial_components(Yr,Cin,fin,[Ain,bin],P,options);

%% update temporal components
P.p = 0;    % set AR temporarily to zero for speed
[C,f,P,S,YrA] = update_temporal_components(Yr,A,b,Cin,fin,P,options);

%% classify components

rval_space = classify_comp_corr(Y,A,C,b,f,options);
ind_corr = rval_space > options.space_thresh;           % components that pass the correlation test
% this test will keep processes

%% further classification with cnn_classifier
try  % matlab 2017b or later is needed
    [ind_cnn,value] = cnn_classifier(A,[d1,d2],'cnn_model',options.cnn_thr);
catch
    ind_cnn = true(size(A,2),1);                        % components that pass the CNN classifier
end

%% event exceptionality

fitness = compute_event_exceptionality(C+YrA,options.N_samples_exc,options.robust_std);
ind_exc = (fitness < options.min_fitness);

%% select components

keep = (ind_corr | ind_cnn) & ind_exc;

%% display kept and discarded components
A_keep = A(:,keep);
C_keep = C(keep,:);
figure;
subplot(121); montage(extract_patch(A(:,keep),[d1,d2],[30,30]),'DisplayRange',[0,0.15]);
title('Kept Components');
subplot(122); montage(extract_patch(A(:,~keep),[d1,d2],[30,30]),'DisplayRange',[0,0.15])
title('Discarded Components');
%% merge found components
[Am,Cm,K_m,merged_ROIs,Pm,Sm] = merge_components(Yr,A_keep,b,C_keep,f,P,S,options);

%%
display_merging = 1; % flag for displaying merging example
if and(display_merging, ~isempty(merged_ROIs))
    i = 1; %randi(length(merged_ROIs));
    ln = length(merged_ROIs{i});
    figure;
    set(gcf,'Position',[300,300,(ln+2)*300,300]);
    for j = 1:ln
        subplot(1,ln+2,j); imagesc(reshape(A_keep(:,merged_ROIs{i}(j)),d1,d2));
        title(sprintf('Component %i',j),'fontsize',16,'fontweight','bold'); axis equal; axis tight;
    end
    subplot(1,ln+2,ln+1); imagesc(reshape(Am(:,K_m-length(merged_ROIs)+i),d1,d2));
    title('Merged Component','fontsize',16,'fontweight','bold');axis equal; axis tight;
    subplot(1,ln+2,ln+2);
    plot(1:T,(diag(max(C_keep(merged_ROIs{i},:),[],2))\C_keep(merged_ROIs{i},:))');
    hold all; plot(1:T,Cm(K_m-length(merged_ROIs)+i,:)/max(Cm(K_m-length(merged_ROIs)+i,:)),'--k')
    title('Temporal Components','fontsize',16,'fontweight','bold')
    drawnow;
end

%% refine estimates excluding rejected components

Pm.p = p;    % restore AR value
[A2,b2,C2] = update_spatial_components(Yr,Cm,f,[Am,b],Pm,options);
[C2,f2,P2,S2,YrA2] = update_temporal_components(Yr,A2,b2,C2,f,Pm,options);


%% do some plotting

[A_or,C_or,S_or,P_or] = order_ROIs(A2,C2,S2,P2); % order components
K_m = size(C_or,1);
[C_df,~] = extract_DF_F(Yr,A_or,C_or,P_or,options); % extract DF/F values (optional)

figure;
subplot(2,2,1)
[Coor,json_file] = plot_contours(A_or,Cn,options,1); % contour plot of spatial footprints
%savejson('jmesh',json_file,'filename');        % optional save json file with component coordinates (requires matlab json library)
fileDir = "/Users/mgs-lab-admin/Desktop/Daniel_Rotation/Data/Daniel_Hao/US_gel/";
filename = "Final_ROIs_" + num2str(K) + "-comp_" + num2str(tau) + "-tau_" + num2str(p) + "-p_" + num2str(min_SNR) + "-min_SNR.fig";
savefig(fileDir + paramName + filename);

%% identify RFP+/- cells
[rfp_positive_inds, rfp_negative_inds] = identifyPositiveCells(zscore_image_rfp, Coor, rfp_thresholds);

% TODO: make sure this is plotting what you'd expect
subplot(2,2,2)
[~,~] = plot_contours(A_or,zscore_image_rfp,options,1); % contour plot of rfp positive

% TODO: make sure this is plotting what you'd expect
subplot(2,2,3)
[rfp_Coor,rfp_contours_json_file] = plot_contours(A_or(:,rfp_positive_inds),image_rfp,options,1); % contour plot of rfp positive

subplot(2,2,4)
[rfp_neg_Coor,rfp_neg_contours_json_file] = plot_contours(A_or(:,rfp_negative_inds),image_rfp,options,1); % contour plot of rfp positive

%% display componentsplot_components_GUI(Yr,A_or,C_or,b2,f2,Cn,options);

%% make movie
if make_movie
    make_patch_video(A_or,C_or,b2,f2,Yr,Coor,options)
end
