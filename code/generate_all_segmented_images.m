% generate_all_segmented_images.m
clc; clear; close all;

img_name = input('Enter image filename (e.g., ''p2.jpg''): ', 's');
a = imread(img_name);
a = imresize(a, [30,30]); % to test quickly on a smaller image

nk = input('Enter the optimal number of clusters: ');

[f, h, ch] = size(a);
n = f * h;
x = double(reshape(a, [n, ch]));

% --- 1. Run K-means ---
fprintf('\nRunning K-means...\n');
[idx_km, ~] = kmeans(x, nk, 'MaxIter', 1000);
img_km = colorize_clusters(x, idx_km, nk, f, h);

% --- 2. Run IMC-1 ---
fprintf('Running IMC-1...\n');
res_imc1 = imc2(x, nk, 1);
img_imc1 = colorize_clusters(x, res_imc1.idx, nk, f, h);

% --- 3. Run IMC-2 ---
fprintf('Running IMC-2...\n');
res_imc2 = imc2(x, nk, (nk/(nk+1)));
img_imc2 = colorize_clusters(x, res_imc2.idx, nk, f, h);

% --- 4. Run SOC ---
fprintf('Running SOC...\n');
fac = factorcal(x, nk, 1);
res_soc = soc(x, nk, fac);
img_soc = colorize_clusters(x, res_soc.idx, nk, f, h);

% --- Display the Results in a Grid ---
figure('Name', '4-Way Segmentation Comparison', 'Position', [100, 100, 1200, 800]);

subplot(2, 3, 1);
imshow(a);
title('Original Image', 'FontSize', 12);

subplot(2, 3, 2);
imshow(img_km);
title('K-Means', 'FontSize', 12);

subplot(2, 3, 3);
imshow(img_imc1);
title('IMC-1', 'FontSize', 12);

subplot(2, 3, 5);
imshow(img_imc2);
title('IMC-2', 'FontSize', 12);

subplot(2, 3, 6);
imshow(img_soc);
title('SOC', 'FontSize', 12);

disp('Visual comparison complete!');


% Helper Function to reconstruct the image with natural cluster average colors
function out_img = colorize_clusters(x, idx, nk, f, h)
    out_img = zeros(f, h, 3, 'uint8');
    mean_colors = zeros(nk, 3);
    
    % Find the average RGB color for each cluster
    for i = 1:nk
        pixels = x(idx == i, :);
        if ~isempty(pixels)
            mean_colors(i, :) = mean(pixels, 1);
        end
    end
    
    % Reconstruct the 2D image matrix
    for i = 1:(f*h)
        [row, col] = ind2sub([f, h], i);
        out_img(row, col, :) = mean_colors(idx(i), :);
    end
end




% saving the generated comparison grid
if builtin('license', 'test', 'Image_Toolbox') && ~isempty(get(0, 'CurrentFigure'))
    % Modern MATLAB recommendation (R2020a and newer) for a crisp, high-res layout
    exportgraphics(gcf, '4_way_segmentation_comparison.png', 'Resolution', 300);
else
    % Fallback option for older MATLAB versions
    saveas(gcf, '4_way_segmentation_comparison.png');
end