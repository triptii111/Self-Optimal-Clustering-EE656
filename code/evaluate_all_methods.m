% evaluate_all_methods.m
clc; clear;

img_name = input('Enter image filename: ', 's');
a = imread(img_name);
a = imresize(a, [30,30]); % for faster testing

nk = input('Enter the optimal number of clusters: ');

[f, h, ch] = size(a);
n = f * h;
x = double(reshape(a, [n, ch]));

% Create a uniform normalized version of x for valid metric evaluation
x_min = min(x);
x_max = max(x);
x_norm = (x - x_min) ./ (x_max - x_min);

fprintf('\n--- Running K-means ---\n');
[idx_km, C_km] = kmeans(x, nk, 'MaxIter', 1000);
s_km = silhouette(x, idx_km);
GSI_km = mean(s_km(~isnan(s_km)));
% Scale K-means centers to [0, 1] for a fair metric comparison
C_km_norm = (C_km - x_min) ./ (x_max - x_min);
[PI_km, SI_km, DI_km] = cluster_metrics(x_norm, idx_km, C_km_norm, nk);

fprintf('--- Running IMC-1 ---\n');
res_imc1 = imc2(x, nk, 1);
s_imc1 = silhouette(x, res_imc1.idx);
[~, GSI_imc1] = slht(s_imc1, res_imc1.idx, res_imc1.n, res_imc1.m, nk);
[PI_imc1, SI_imc1, DI_imc1] = cluster_metrics(x_norm, res_imc1.idx, res_imc1.cc_norm, nk);

fprintf('--- Running IMC-2 ---\n');
res_imc2 = imc2(x, nk, (nk/(nk+1)));
s_imc2 = silhouette(x, res_imc2.idx);
[~, GSI_imc2] = slht(s_imc2, res_imc2.idx, res_imc2.n, res_imc2.m, nk);
[PI_imc2, SI_imc2, DI_imc2] = cluster_metrics(x_norm, res_imc2.idx, res_imc2.cc_norm, nk);

fprintf('--- Running SOC ---\n');
fac = factorcal(x, nk, 1);
res_soc = soc(x, nk, fac);
s_soc = silhouette(x, res_soc.idx);
[~, GSI_soc] = slht(s_soc, res_soc.idx, res_soc.n, res_soc.m, nk);
[PI_soc, SI_soc, DI_soc] = cluster_metrics(x_norm, res_soc.idx, res_soc.cc_norm, nk);

% Display Final Table
fprintf('\n================ RESULTS TABLE ================\n');
fprintf('%-10s | %-8s | %-8s | %-8s | %-8s\n', 'Method', 'GSI', 'PI', 'SI', 'DI');
fprintf('-----------------------------------------------\n');
fprintf('%-10s | %-8.4f | %-8.4f | %-8.4f | %-8.4f\n', 'K-means', GSI_km, PI_km, SI_km, DI_km);
fprintf('%-10s | %-8.4f | %-8.4f | %-8.4f | %-8.4f\n', 'IMC-1', GSI_imc1, PI_imc1, SI_imc1, DI_imc1);
fprintf('%-10s | %-8.4f | %-8.4f | %-8.4f | %-8.4f\n', 'IMC-2', GSI_imc2, PI_imc2, SI_imc2, DI_imc2);
fprintf('%-10s | %-8.4f | %-8.4f | %-8.4f | %-8.4f\n', 'SOC', GSI_soc, PI_soc, SI_soc, DI_soc);
fprintf('===============================================\n');