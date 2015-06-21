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
config.init = 0;
I = im2double(imread('applications/deep_edge_aware_filters/images/1.png'));



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% bilateral filter, sigma_s = 7, sigma_r = 0.1
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%model_path = 'applications/deep_edge_aware_filters/models/bilateral.mat';
%beta = 8.388608e+02 / 7;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% L0 smooth filter, lambda = 0.02, kappa default
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
model_path = 'applications/deep_edge_aware_filters/models/L0_smooth.mat';
beta = 8.388608e+03 / 2;


fprintf('preparing the network...\n');
prepare_net_filter(size(I, 1), size(I, 2), model_path);

fprintf('filtering the image...\n');
tic
S = I;
betamax = 1e5;
fx = [1, -1];
fy = [1; -1];
[N,M,D] = size(I);
sizeI2D = [N,M];
otfFx = psf2otf(fx,sizeI2D);
otfFy = psf2otf(fy,sizeI2D);
Normin1 = fft2(S);
Denormin2 = abs(otfFx).^2 + abs(otfFy ).^2;
Denormin2 = repmat(Denormin2,[1,1,D]);

Denormin   = 1 + beta*Denormin2;
h_input = [diff(S,1,2), S(:,1,:) - S(:,end,:)];
v_input = [diff(S,1,1); S(1,:,:) - S(end,:,:)];
h_input = h_input * 2;
v_input = v_input * 2;
v_input = config.NEW_MEM(v_input);
h_input = config.NEW_MEM(h_input);

out = apply_net_filter(v_input, h_input);

v = out(:,:,:,1);
h = out(:,:,:,2);
v = v / 2;
h = h / 2;
h(:, end, :) = S(:,1,:) - S(:,end,:);
v(end, :, :) = S(1,:,:) - S(end,:,:);

Normin2 = [h(:,end,:) - h(:, 1,:), -diff(h,1,2)];
Normin2 = Normin2 + [v(end,:,:) - v(1, :,:); -diff(v,1,1)];
FS = (Normin1 + beta*fft2(Normin2))./Denormin;
filtered = real(ifft2(FS));
toc

figure;
imshow([I, filtered]); drawnow();



