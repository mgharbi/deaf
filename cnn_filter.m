
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

global config mem;
model_path = 'applications/deep_edge_aware_filters/models/L0_smooth.mat';

h = 64;
w = 64;
prepare_net_filter(w,h,model_path);


nlayers = 3;
config.weights{3+nlayers} = 0;


% Gen out
I = imread('data/debug_input.png');
I = im2double(I);
input_piece = repmat(I,[1,1,1,2]);
size(input_piece)
input_piece = config.NEW_MEM(input_piece);
op_test_pipe(input_piece, mem.fake_output_for_test);
output_piece = gather(mem.output);
input_piece = gather(input_piece);

c1      = gather(config.weights{1});
c1_b    = gather(config.weights{1+nlayers});
c2      = gather(config.weights{2});
c2_b    = gather(config.weights{2+nlayers});
d       = gather(config.weights{3});
d_b     = gather(config.weights{3+nlayers});

outpath = fullfile('output', 'weights.mat');
save(outpath,'c1','c1_b','c2','c2_b','d','d_b','-v7');

a_c1 = gather(mem.activations{1});
a_c2 = gather(mem.activations{2});
a_d = gather(mem.activations{3});
outpath = fullfile('output', 'activations.mat');
save(outpath,'a_c1','a_c2','a_d','-v7');

outpath = fullfile('output', 'rand_out.mat');
save(outpath,'input_piece', 'output_piece','-v7');
