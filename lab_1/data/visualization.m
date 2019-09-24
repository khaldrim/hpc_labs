# Abrir la image

# Open and reshape the image to a 256 x 256 matrix
f = fopen('circulos_secuential.raw', 'r');
s = fread(f, 'int');
fclose(f);
s = reshape(s, 256, 256);
s = s';

# See the image
imagesc(s);
axis('square');
colormap(gray);