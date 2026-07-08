% find_optimal_clusters.m
clc; clear; close all;

img_name = input('Enter image filename (e.g., ''p2.jpg''): ', 's');
a = imread(img_name);

% Optional: Downsampling image for faster testing 
a = imresize(a, [30,30]); 

[f, h, ch] = size(a);
n = f * h;
x = double(reshape(a, [n, ch]));

max_clusters = 5; % Evaluates M=2 through M=5
GSI_values = zeros(1, max_clusters);

disp('Evaluating optimal cluster count...');

for nk = 2:max_clusters
    fprintf('Running SOC for %d clusters...\n', nk);
    
    fac = factorcal(x, nk, 1);
    result = soc(x, nk, fac);
    
    s = silhouette(x, result.idx);
    [~, GSI_values(nk)] = slht(s, result.idx, result.n, result.m, nk);
end

% Plotting the GSI curve
figure('Name', 'Optimal Number of Clusters');
plot(2:max_clusters, GSI_values(2:end), '--bs', 'LineWidth', 2, 'MarkerFaceColor', 'k');
xlabel('Number of clusters');
ylabel('Global Silhouette Index (GSI)');
title('Variation of GSI with number of clusters');
grid on;

[max_gsi, optimal_idx] = max(GSI_values(2:end));
optimal_k = optimal_idx + 1;
fprintf('\n>>> The optimal number of clusters is: %d (GSI = %.4f) <<<\n', optimal_k, max_gsi);