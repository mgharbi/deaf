addpath applications/deep_edge_aware_filters/utility/GT_filters/
addpath applications/deep_edge_aware_filters/utility/GT_filters/L0smoothing/
addpath data/

clear;
p_sz = 64;
step = p_sz;
listing = dir('data/deepeaf/BSDS500/*.tif');
for m = 1 : 1
    I = imread(strcat('data/deepeaf/BSDS500/', listing(m).name));
    [h,w,~] = size(I);
    n_patch_x = int32((w - p_sz)/step);
    n_patch_y = int32((h - p_sz)/step);
    num_patches = n_patch_x*n_patch_y;

    idx_x = randperm(n_patch_x);
    idx_y = randperm(n_patch_y);

    samples = zeros(p_sz, p_sz, 3, num_patches);
    labels = zeros(size(samples));
    idx = 1;
    for i = idx_y
        first_i = (i-1)*p_sz+1;
        for j = idx_x
            first_j            = (j-1)*p_sz+1;
            patch              = im2double(I(first_i:first_i+p_sz-1, first_j:first_j+p_sz-1, :));
            patch_filtered     = GT_filter(patch);
            samples(:,:,:,idx) = patch;
            labels(:,:,:,idx)  = patch_filtered;
            idx                = idx + 1;
        end
    end

    samples = single(samples);
    labels  = single(labels);
    % save it
    filename = strcat('data/deepeaf/L0/train/patches_', num2str(m));
    save(filename, '-v7.3', 'samples', 'labels');
end

