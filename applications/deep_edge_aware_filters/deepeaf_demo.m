%--------------------------------------------------------------------------------------------------------
% The system is created based on the principles described in the following papers
% [1] Li Xu, Jimmy SJ. Ren, Qiong Yan, Renjie Liao, Jiaya Jia, "Deep Edge-Aware Filters", 
% The 32nd International Conference on Machine Learning (ICML 2015). Lille, France, July 6-11, 2015
% [2] Jimmy SJ. Ren and Li Xu, "On Vectorization of Deep Convolutional Neural Networks for Vision Tasks", 
% The 29th AAAI Conference on Artificial Intelligence (AAAI-15). Austin, Texas, USA, January 25-30, 2015
%--------------------------------------------------------------------------------------------------------

addpath applications/deep_edge_aware_filters/
addpath applications/deep_edge_aware_filters/utility/
addpath applications/deep_edge_aware_filters/models/
addpath applications/deep_edge_aware_filters/images/
addpath utils/
addpath cuda/
addpath mem/
addpath layers/
addpath layers_adapters/
addpath pipeline/

global config;
% load the image you like
I = im2double(imread('applications/deep_edge_aware_filters/images/1.png'));

% to switch among filters, just comment out the previous 'model_path' and 'beta' and
% uncomment the new ones

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% L0 smooth filter, lambda = 0.02, kappa default
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
model_path = 'applications/deep_edge_aware_filters/models/L0_smooth.mat';
beta = 8.388608e+03 / 2;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% bilateral filter, sigma_s = 7, sigma_r = 0.1
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% model_path = 'applications/deep_edge_aware_filters/models/bilateral.mat';
% beta = 8.388608e+02 / 7;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Region cov filter, radius = 10, ps = 4, sigma = 0.2, model = 1 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% model_path = 'applications/deep_edge_aware_filters/models/regcov.mat';
% beta = 8.388608e+02 / 5;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% photoshop facet filter
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% model_path = 'applications/deep_edge_aware_filters/models/ps-facet.mat';
% beta = 8.388608e+02 / 5;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% shock filter, dt=0.1; h=1; iter=30; 'org'
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% model_path = 'applications/deep_edge_aware_filters/models/shock.mat';
% beta = 8.388608e+02 / 5;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% tsmooth filter, lambda=0.01, sigma=3
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% model_path = 'applications/deep_edge_aware_filters/models/tsmooth.mat';
% beta = 8.388608e+02 / 5;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% iterative bilateral filter, sigma_s = 7, sigma_r = 0.1, iter = 3
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% model_path = 'applications/deep_edge_aware_filters/models/iterative_bilateral.mat';
% beta = 8.388608e+02 / 5;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% local laplacian filter, delta = 0.4, alpha = 2
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% model_path = 'applications/deep_edge_aware_filters/models/local_lap_smooth.mat';
% beta = 8.388608e+03 / 2;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% local laplacian filter, delta = 0.4, alpha = 0.25
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% model_path = 'applications/deep_edge_aware_filters/models/local_lap_enhance.mat';
% beta = 8.388608e+03 / 2;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% weighted median filter, 10, 25.5, 256, 256, 1, 'exp'
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% model_path = 'applications/deep_edge_aware_filters/models/wmf.mat';
% beta = 8.388608e+03 / 2;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% WLS, default parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% model_path = 'applications/deep_edge_aware_filters/models/wls.mat';
% beta = 8.388608e+02 / 5;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% rolling guidance filter, default parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% model_path = 'applications/deep_edge_aware_filters/models/rolling.mat';
% beta = 8.388608e+03 / 2;


fprintf('preparing the network...\n');
prepare_net_filter(size(I, 1), size(I, 2), model_path);

fprintf('filtering the image...\n');
S         = I;
betamax   = 1e5;
fx        = [1, -1];
fy        = [1; -1];
[N,M,D]   = size(I);
sizeI2D   = [N,M];
otfFx     = psf2otf(fx,sizeI2D);
otfFy     = psf2otf(fy,sizeI2D);
Normin1   = fft2(S);
Denormin2 = abs(otfFx).^2 + abs(otfFy ).^2;
Denormin2 = repmat(Denormin2,[1,1,D]);

Denormin = 1 + beta*Denormin2;
h_input  = [diff(S,1,2), S(:,1,:) - S(:,end,:)];
v_input  = [diff(S,1,1); S(1,:,:) - S(end,:,:)];
h_input  = h_input * 2;
v_input  = v_input * 2;
v_input  = config.NEW_MEM(v_input);
h_input  = config.NEW_MEM(h_input);

tic
out = apply_net_filter(v_input, h_input);
toc

figure;
imshow(squeeze(out(:,:,:,1)));

v            = out(:,:,:,1);
h            = out(:,:,:,2);
v            = v / 2;
h            = h / 2;
h(:, end, :) = S(:,1,:) - S(:,end,:);
v(end, :, :) = S(1,:,:) - S(end,:,:);

Normin2  = [h(:,end,:) - h(:, 1,:), -diff(h,1,2)];
Normin2  = Normin2 + [v(end,:,:) - v(1, :,:); -diff(v,1,1)];
FS       = (Normin1 + beta*fft2(Normin2))./Denormin;
filtered = real(ifft2(FS));

figure;
imshow([I, filtered]); drawnow();



