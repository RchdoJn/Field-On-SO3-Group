function [pcd1, pcd2, alpha, data] = InitializeNFieldComplex(X, T, FixedTriangle, FixedDirection, Nfold)
% InitializeCrossFieldComplex4 solves a relaxed N-fold complex field
% initialization, then extracts alpha = arg(z) / Nfold.

if nargin < 3 || isempty(FixedTriangle)
    FixedTriangle = [];
end
if nargin < 4 || isempty(FixedDirection)
    FixedDirection = [];
end
if nargin < 5 || isempty(Nfold)
    Nfold = 4;
end
validateattributes(Nfold, {'numeric'}, {'scalar', 'integer', '>=', 1});

[fn, Q, b1, b2] = localFaceFrames(X, T);
[E, Ine, adjmat, Rij] = localConnectionData(X, T, fn);
nf = size(T, 1);
ne = size(E, 1);
w = edge_weight(X, T, E);

beta = zeros(ne, 1);
rows = zeros(numel(Ine), 1);
colsF = zeros(numel(Ine), 1);
colsG = zeros(numel(Ine), 1);
valsF = complex(zeros(numel(Ine), 1));
valsG = complex(zeros(numel(Ine), 1));
rhs = complex(zeros(numel(Ine), 1));

for k = 1:numel(Ine)
    ind = Ine(k);
    i = E(ind, 1);
    j = E(ind, 2);
    f = adjmat(i, j);
    g = adjmat(j, i);

    transported = (Rij(3*ind-2:3*ind, :) * b1(f, :)')';
    beta(ind) = atan2(dot(transported, b2(g, :)), dot(transported, b1(g, :)));

    rows(k) = k;
    colsF(k) = f;
    colsG(k) = g;
    sw = sqrt(max(w(ind), 1e-12));
    valsF(k) = -sw * exp(1i * Nfold * beta(ind));
    valsG(k) = sw;
end

A = sparse([rows; rows], [colsF; colsG], [valsF; valsG], numel(Ine), nf);

[fixedMask, fixedAlpha] = localFixedAngles(FixedTriangle, FixedDirection, fn, b1, b2, nf);
if ~any(fixedMask)
    fixedMask(1) = true;
    fixedAlpha(1) = 0;
end

fixed = find(fixedMask);
free = find(~fixedMask);
z = complex(zeros(nf, 1));
z(fixed) = exp(1i * Nfold * fixedAlpha(fixed));

rhsFree = rhs - A(:, fixed) * z(fixed);
z(free) = A(:, free) \ rhsFree;

zRaw = z;
zRawAbs = abs(zRaw);
finiteRawAbs = zRawAbs(isfinite(zRawAbs));
if isempty(finiteRawAbs)
    minRawVectorNorm = NaN;
else
    minRawVectorNorm = min(finiteRawAbs);
end

zAbsForNormalize = zRawAbs;
zAbsForNormalize(zAbsForNormalize < 1e-12) = 1;
z = z ./ zAbsForNormalize;

alpha = angle(z) / Nfold;
[pcd1, pcd2] = localFieldFromAngles(alpha, b1, b2);

data = struct();
data.method = sprintf('%d-fold complex Laplacian', Nfold);
data.Nfold = Nfold;
data.fn = fn;
data.Q = Q;
data.b1 = b1;
data.b2 = b2;
data.edges = E;
data.interiorEdges = Ine;
data.adjmat = adjmat;
data.Rij = Rij;
data.beta = beta;
data.zRaw = zRaw;
data.zRawAbs = zRawAbs;
data.minRawVectorNorm = minRawVectorNorm;
data.z = z;
data.zAbs = abs(z);
data.alpha = alpha;
data.fixedMask = fixedMask;
data.fixedAlpha = fixedAlpha;
data.energy = localCrossEnergy(alpha, beta, Ine, E, adjmat, w, Nfold);
end

function [fn, Q, b1, b2] = localFaceFrames(X, T)
nf = size(T, 1);
v1 = X(T(:,2), :) - X(T(:,1), :);
v2 = X(T(:,3), :) - X(T(:,1), :);
fn = cross(v1, v2, 2);
fn = fn ./ max(1e-12, sqrt(sum(fn.^2, 2)));

z = [0; 0; 1];
Q = zeros(3, 3, nf);
b1 = zeros(nf, 3);
b2 = zeros(nf, 3);
for f = 1:nf
    n = fn(f, :)';
    n = n / max(norm(n), 1e-12);
    c = max(-1, min(1, z' * n));

    if c > 1 - 1e-12
        R = eye(3);
    elseif c < -1 + 1e-12
        % pi rotation: choose x-axis, which is orthogonal to the source z-axis.
        axis = [1; 0; 0];
        K = local_hat(axis);
        R = eye(3) + 2 * (K * K);
    else
        axis = cross(z, n);
        s = norm(axis);
        axis = axis / max(s, 1e-12);
        theta = atan2(s, c);
        K = local_hat(axis);
        R = eye(3) + sin(theta) * K + (1 - cos(theta)) * (K * K);
    end

    tangent1 = R(:, 1) - n * (n' * R(:, 1));
    if norm(tangent1) < 1e-12
        axis = [1; 0; 0];
        if abs(axis' * n) > 0.9
            axis = [0; 1; 0];
        end
        tangent1 = axis - n * (n' * axis);
    end
    tangent1 = tangent1 / max(norm(tangent1), 1e-12);
    tangent2 = cross(n, tangent1);
    tangent2 = tangent2 / max(norm(tangent2), 1e-12);

    Q(:,:,f) = [tangent1, tangent2, n];
    b1(f, :) = tangent1';
    b2(f, :) = tangent2';
end
end

function K = local_hat(v)
K = [0, -v(3), v(2); v(3), 0, -v(1); -v(2), v(1), 0];
end

function [E, Ine, adjmat, Rij] = localConnectionData(X, T, fn)
nv = size(X, 1);
nf = size(T, 1);
hat = @(x) [0, -x(3), x(2); x(3), 0, -x(1); -x(2), x(1), 0];

E = unique(sort(reshape([T, T(:, [2 3 1])], [], 2), 2), 'rows');
ne = size(E, 1);
adjmat = sparse(T, T(:, [2 3 1]), repmat(1:nf, 1, 3), nv, nv);

isInterior = false(ne, 1);
Rij = zeros(3*ne, 3);
for ind = 1:ne
    i = E(ind, 1);
    j = E(ind, 2);
    f = adjmat(i, j);
    g = adjmat(j, i);
    isInterior(ind) = f > 0 && g > 0;
    if ~isInterior(ind)
        continue;
    end

    eij = X(j, :) - X(i, :);
    eij = eij ./ max(norm(eij), 1e-12);
    ang = atan2(cross(fn(f, :), fn(g, :)) * eij', fn(f, :) * fn(g, :)');
    nh = hat(eij);
    Rij(3*ind-2:3*ind, :) = eye(3) + sin(ang) * nh + (1-cos(ang)) * (nh * nh);
end
Ine = find(isInterior);
end

function [fixedMask, fixedAlpha] = localFixedAngles(FixedTriangle, FixedDirection, fn, b1, b2, nf)
fixedMask = false(nf, 1);
fixedAlpha = zeros(nf, 1);
for k = 1:numel(FixedTriangle)
    f = FixedTriangle(k);
    d = FixedDirection(k, :);
    d = d - dot(d, fn(f, :)) * fn(f, :);
    if norm(d) < 1e-12
        d = b1(f, :);
    else
        d = d ./ norm(d);
    end
    fixedMask(f) = true;
    fixedAlpha(f) = atan2(dot(d, b2(f, :)), dot(d, b1(f, :)));
end
end

function [pcd1, pcd2] = localFieldFromAngles(alpha, b1, b2)
pcd1 = cos(alpha) .* b1 + sin(alpha) .* b2;
pcd2 = -sin(alpha) .* b1 + cos(alpha) .* b2;

% Enforce unit-length tangent directions per face (robust to numeric drift).
n1 = sqrt(sum(pcd1.^2, 2));
n2 = sqrt(sum(pcd2.^2, 2));
pcd1 = pcd1 ./ max(n1, 1e-12);
pcd2 = pcd2 ./ max(n2, 1e-12);
end

function energy = localCrossEnergy(alpha, beta, Ine, E, adjmat, w, Nfold)
energy = 0;
for ind = Ine'
    i = E(ind, 1);
    j = E(ind, 2);
    f = adjmat(i, j);
    g = adjmat(j, i);
    diff = alpha(g) - alpha(f) - beta(ind);
    energy = energy + w(ind) * (1 - cos(Nfold*diff));
end
end
