function gen_output_from_conv_f()
    global config mem;

    % config.SCALE_OUTPUT();  % this scaling may not be necessary in the future

    % mem.output = reshape(accumarray(mem.gen_out_matrix, mem.activations{length(mem.activations)}(:)), size(mem.output));
    mem.output = bsxfun(@times, reshape(accumarray(mem.gen_out_matrix, mem.activations{length(mem.activations)}(:)), size(mem.output)), mem.one_over_add_counts);
end


