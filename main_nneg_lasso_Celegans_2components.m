close all
clear;
clc;

%% Load pure chemicals
rpf =(2994-2913)/(76-40); % Use DMSO to calibrate
Raman_shift = linspace(2913-40*rpf, 2994+22*rpf,100);

load 'Data/Pure_chemicals.mat'

% Normalize TAG and BSA
BSA_n = ( BSA - min(BSA) ) ./ (max(BSA) - min(BSA));
TAG_n = ( TAG - min(TAG) ) ./ (max(TAG) - min(TAG));

figure;
plot(Raman_shift, BSA_n, 'Linewidth',1);
hold on
plot(Raman_shift, TAG_n, 'Linewidth',1);
hold off
legend('Protein (BSA)','Lipid (TAG)')

%% Load data
filepath = 'Data/';
filename = 'Celegans_1040-100_800-20';
ext = '.txt';

% load txt image
raw_data = importdata([filepath, filename ext]);

Nx = 400;
Ny = 400;
Nz = 100;

y = reshape(raw_data, [Nx,Ny,Nz]);


%% Find threshold value for background & Subtract background

% Histogram all pixel values to help choose the background value
y_sum = squeeze(mean(y,3));
figure; histogram(y_sum);title('Histogram of all pixel values')

% Set the threshold value and plot bkg mask
bkg_threshold = 0.00;

BGmask=zeros(size(y_sum));
BG=find(y_sum < bkg_threshold);
BGmask(BG)=1;
figure;imagesc(BGmask);title('Background mask')

% Subtrack background from the raw image
BG_spectrum = zeros(Nz,1);
for i=1:1:Nz
    y_temp = y(:,:,i);
    BG_spectrum(i) = mean(y_temp(BGmask == 1)); 
end
BG_spectrum = BG_spectrum';
figure;plot(BG_spectrum);title('Background spectrum')

y_sub = zeros(size(y));
for i=1:1:Nz
    y_sub(:,:,i) = y(:,:,i) - BG_spectrum(i);
end

%% nonnegative LASSO unmixing, tuning L for sparsity levels
% Need parallel computing toolbox
k    = 2;                 % Set number of components to 2
ref  = [BSA_n,TAG_n];     % Generate spectral reference matrix
L    = 5e-2;              % Set sparsity level (\lambda), if 0 then LS fitting
a    = 1;                 % ADMM variable to control convergence speed, default 1
iter = 5;                 % Number of iterations for ADMM
tic
C    = nneg_lasso( y, ref, L, a, iter);
toc
%% Quick check unmixing quality, modify as needed

disp_min = prctile(reshape(C(:,:,1),[Nx*Ny,1]),0.4);
disp_max = prctile(reshape(C(:,:,1),[Nx*Ny,1]),99.7);

figure;
clims = [disp_min disp_max];
subplot(1,2,1);imagesc(C(:,:,1),clims); colormap bone; axis off; axis square
subplot(1,2,2);imagesc(C(:,:,2),clims); colormap bone; axis off; axis square

%% Output as txt file, modify as needed

opt_filepath = 'chemical_maps/';

output_ext   = '.txt';
Protein_map  = C(:,:,1);
TAG_map      = C(:,:,2);

protein_out_filename = ['Protein_lambda_', num2str(L) '_',filename, '_Protein', output_ext];
TAG_out_filename = ['Lipid_lambda_', num2str(L) '_',filename, '_TAG', output_ext];

dlmwrite([opt_filepath, protein_out_filename], Protein_map, 'delimiter','\t');
dlmwrite([opt_filepath, TAG_out_filename], TAG_map, 'delimiter','\t');
