%f = fopen('results/secuential.raw', 'r');
%M = fread(f, 'double');
%M = reshape(M, 2001, 2001);
%M = M'
%imagesc(M);
%colormap([jet(); flipud(jet()); 0 0 0]);
%clear();

f = fopen('results/parallel.raw', 'r');
M = fread(f, 'double');
M = reshape(M, 6001, 6001);
M = M'
imagesc(M);
colormap([jet(); flipud(jet()); 0 0 0]);
clear();