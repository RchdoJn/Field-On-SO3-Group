function edgeWeights = edge_weight(X, T, E)
%EDGE_WEIGHT Compute intrinsic edge weights w_fg = |e_fg| / |c_f-c_g| for interior edges.
% Boundary edges are assigned zero weight.

nv = size(X, 1);
nf = size(T, 1);
ne = size(E, 1);

adjmat = sparse(T, T(:, [2 3 1]), repmat(1:nf, 1, 3), nv, nv);
centers = (X(T(:,1),:) + X(T(:,2),:) + X(T(:,3),:)) / 3;

edgeWeights = zeros(ne, 1);
for ind = 1:ne
    i = E(ind, 1);
    j = E(ind, 2);
    f = adjmat(i, j);
    g = adjmat(j, i);
    if f == 0 || g == 0
        edgeWeights(ind) = 0;
        continue;
    end

    elen = norm(X(i,:) - X(j,:));
    clen = norm(centers(f,:) - centers(g,:));
    edgeWeights(ind) = elen / max(clen, 1e-12);
end
end