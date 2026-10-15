function [kf1, kf2, pcdf1, pcdf2] = PrincipalDirectFace(x, t)
% estimute on vertex' pcd firstly, and then average on faces
nv = size(x ,1);
nf = size(t, 1);
% --- face normals & face areas
v1 = x(t(:,2),:) - x(t(:,1),:);
v2 = x(t(:,3),:) - x(t(:,1),:);
fn = cross(v1, v2, 2);
faceArea = 0.5 * sqrt(sum(fn.^2, 2));
fn = fn ./ max(1e-12, sqrt(sum(fn.^2,2))); % unit face normals

% --- vertex normals (area-weighted)
vn = zeros(nv,3);
for i = 1:nf
    for k = 1:3
        vn(t(i,k),:) = vn(t(i,k),:) + fn(i,:) * faceArea(i);
    end
end
vn = vn ./ max(1e-12, sqrt(sum(vn.^2,2)));

raxis = zeros(nv, 3);
iaxis = zeros(nv, 3);
for i = 1:nv
    tmp = [1 0 0];
    n = vn(i,:);
    if abs(dot(tmp,n)) > 0.9
        tmp = [0 1 0];
    end
    ra = tmp - dot(tmp,n)*n;
    ra = ra / norm(ra);
    ia = cross(n,ra);
    raxis(i, :) = ra;
    iaxis(i, :) = ia;
end

adj = sparse(t, t(:, [2 3 1]), 1, nv, nv);

k1 = zeros(nv,1); k2 = zeros(nv,1);
pcd1 = zeros(nv,3); pcd2 = zeros(nv,3);
for i = 1:nv
    r1 = find(adj(:,i));
    e = x(r1, :) - x(i, :);
    ra = raxis(i, :);
    ia = iaxis(i, :);
    t1 = e * ra';
    t2 = e * ia';
    et = [t1, t2];
    n1 = vn(r1, :) - vn(i, :);
    n2 = n1 * ra';
    n3 = n1 * ia';
    n = [n2, n3];
    k1 = -sum(n.*et, 2)./sum(et.*et, 2);
    r = size(e, 1);
    K = zeros(2,2);
    for j = 1:r
        K = K + (et(j, :)'.*et(j, :)).*k1(j, :);
    end
    [V, D] = eig(K);
    eigv = diag(D);
    [eigv_sort, idx] = sort(eigv, 'descend');
    v_sort = V(:, idx);
    % map eigenvectors back to 3D
    pdir1 = v_sort(1, 1) * ra + v_sort(2, 1) * ia; % careful with transpose
    pdir2 = v_sort(1, 2) * ra + v_sort(2, 2) * ia;
    % normalize
    pdir1 = pdir1 / norm(pdir1);
    pdir2 = pdir2 / norm(pdir2);

    k1(i) = eigv_sort(1);
    k2(i) = eigv_sort(2);
    pcd1(i,:) = pdir1;
    pcd2(i,:) = pdir2;

end

pcdf1 = pcd1(t(:,1), :) + pcd1(t(:,2), :) + pcd1(t(:,3), :);
pcdf2 = pcd2(t(:,1), :) + pcd2(t(:,2), :) + pcd2(t(:,3), :);
pro = @(x, f) x - sum(x.*f, 2).*f;
pcdf1 = pro(pcdf1, fn);
pcdf2 = pro(pcdf2, fn);
pcdf1 = pcdf1./sqrt(sum(pcdf1.^2, 2));
pcdf2 = pcdf2./sqrt(sum(pcdf2.^2, 2));
pcdf2 = cross(fn, pcdf1);
kf1 = (k1(t(:, 1), :) + k1(t(:, 2), :) + k1(t(:, 3), :))./3;
kf2 = (k2(t(:, 1), :) + k2(t(:, 2), :) + k2(t(:, 3), :))./3;

end