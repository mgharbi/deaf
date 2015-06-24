model = load('output/epoc0.mat');
model = model.model;
model

conv1 = gather(model.weights{1});
conv1_sz = model.kernel_size(1,:); conv1_sz = [conv1_sz 3];

conv1 = reshape(conv1, [size(conv1,1), conv1_sz]);
mini = min(conv1(:));
maxi = max(conv1(:));
conv1 = conv1-mini;
conv1 = conv1/(maxi-mini);
conv1 = uint8(conv1*255);
conv1 = permute(conv1, [2 3 4 1]);
size(conv1)

m = montage(conv1);
% imshow(squeeze(conv1(1,:,:,:)));

