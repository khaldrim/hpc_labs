f = fopen('results/secuential.raw', 'r');
M = fread(f, 'double');
M = reshape(M, 2000, 2000);
imagesc(M);
colormap([jet(); flipud(jet()); 0 0 0]);