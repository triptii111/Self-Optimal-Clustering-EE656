% cluster_metrics.m
function [PI, SI, DI] = cluster_metrics(x, idx, centers, nk)
    n = size(x, 1);
    
    intra_dist_sum = 0;
    max_dia = 0;
    
    % Compute intra-cluster variances and diameters
    for i = 1:nk
        pts = x(idx == i, :);
        if isempty(pts)
            continue; 
        end
        
        % For PI and SI numerator
        diffs = pts - repmat(centers(i,:), size(pts,1), 1);
        sq_dists = sum(diffs.^2, 2);
        intra_dist_sum = intra_dist_sum + sum(sq_dists);
        
        % For DI denominator: Approximate cluster diameter to save memory on large images
        cluster_dia = max(sqrt(sq_dists)) * 2; 
        max_dia = max(max_dia, cluster_dia);
    end
    
    % Compute inter-cluster center distances
    center_dists = pdist(centers); % Pairwise distances between centers
    sum_center_dists_sq = sum(center_dists.^2);
    min_center_dist_sq = min(center_dists.^2);
    min_center_dist = min(center_dists);
    
    % PI and SI calculations
    PI = intra_dist_sum / (n * sum_center_dists_sq);
    SI = intra_dist_sum / (n * min_center_dist_sq);
    
    % DI calculation
    if max_dia == 0
        DI = inf;
    else
        DI = min_center_dist / max_dia;
    end
end