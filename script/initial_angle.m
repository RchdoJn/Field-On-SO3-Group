function [theta0, frameData] = initial_angle(X, T, E, adjmat, fn, Rij, FixedTriangle, FixedDirection, Nfold)
%INITIAL_ANGLE Compute v2 initialization by solving psi = N*theta linear system.

if nargin < 7 || isempty(FixedTriangle)
    FixedTriangle = [];
end
if nargin < 8 || isempty(FixedDirection)
    FixedDirection = [];
end
if nargin < 9 || isempty(Nfold)
    Nfold = 4;
end

nf = size(T, 1);
ne = size(E, 1);

[Q, b1, b2] = local_minimal_Q_frames(fn);

w = edge_weight(X, T, E);
L = spalloc(nf, nf, 4 * ne);
alpha = zeros(ne, 1);

for ind = 1:ne
    i = E(ind, 1);
    j = E(ind, 2);
    f = adjmat(i, j);
    g = adjmat(j, i);
    if f == 0 || g == 0
        continue;
    end

    S0 = Rij(3*ind-2:3*ind, :) * Q(:,:,f) * Q(:,:,g)';
    phi0 = logSO3_vee_local(S0);
    alpha(ind) = fn(g, :) * phi0;

    wij = w(ind);

    L(f, f) = L(f, f) + wij;
    L(g, g) = L(g, g) + wij;
    L(f, g) = L(f, g) - wij;
    L(g, f) = L(g, f) - wij;

end

fixedMask = false(nf, 1);
thetaFixed = zeros(nf, 1);
psiFixed = zeros(nf, 1);
for k = 1:numel(FixedTriangle)
    f = FixedTriangle(k);
    dir = FixedDirection(k, :);
    dir = dir - dot(dir, fn(f,:)) * fn(f,:);
    if norm(dir) < 1e-12
        dir = b1(f, :);
    else
        dir = dir / norm(dir);
    end
    thetaBar = atan2(dot(dir, b2(f,:)), dot(dir, b1(f,:)));
    fixedMask(f) = true;
    thetaFixed(f) = wrap_to_nrosy_theta(thetaBar, Nfold);
    psiFixed(f) = wrap_to_pi(Nfold * thetaFixed(f));
end

if ~any(fixedMask)
    fixedMask(1) = true;
    thetaFixed(1) = 0;
    psiFixed(1) = 0;
end

free = find(~fixedMask);
fixed = find(fixedMask);
psi = zeros(nf, 1);
psi(fixed) = psiFixed(fixed);

% Periodic-aware initialization:
% 1) Start from principal beta on each edge.
% 2) Iteratively lift beta by 2*pi*k (phase unwrapping) using current psi.
% 3) Re-solve sparse linear system to better match periodic energy.
beta0 = wrap_to_pi(Nfold * alpha);
betaLift = beta0;

d = build_rhs_from_beta(betaLift, E, adjmat, w, ne, nf);
rhs = d(free) - L(free, fixed) * psi(fixed);
psi(free) = L(free, free) \ rhs;

unwrapIters = 1;
for it = 1:unwrapIters
    for ind = 1:ne
        i = E(ind, 1);
        j = E(ind, 2);
        f = adjmat(i, j);
        g = adjmat(j, i);
        if f == 0 || g == 0
            continue;
        end
        val = psi(f) - psi(g) + beta0(ind);
        k = round(val / (2*pi));
        betaLift(ind) = beta0(ind) - 2*pi*k;
    end

    d = build_rhs_from_beta(betaLift, E, adjmat, w, ne, nf);
    rhs = d(free) - L(free, fixed) * psi(fixed);
    psi(free) = L(free, free) \ rhs;
end

psi = wrap_to_pi(psi);
theta0 = psi / Nfold;

frameData = struct();
frameData.b1 = b1;
frameData.b2 = b2;
frameData.Q = Q;
frameData.alpha = alpha;
frameData.beta0 = beta0;
frameData.betaLift = betaLift;
frameData.weights = w;
frameData.fixedMask = fixedMask;
frameData.thetaFixed = thetaFixed;
frameData.psiFixed = psiFixed;
frameData.psi = psi;
end

function d = build_rhs_from_beta(beta, E, adjmat, w, ne, nf)
d = zeros(nf, 1);
for ind = 1:ne
    i = E(ind, 1);
    j = E(ind, 2);
    f = adjmat(i, j);
    g = adjmat(j, i);
    if f == 0 || g == 0
        continue;
    end
    wij = w(ind);
    b = beta(ind);
    d(f) = d(f) - wij * b;
    d(g) = d(g) + wij * b;
end
end

function x = wrap_to_pi(x)
x = mod(x + pi, 2*pi) - pi;
tol = 1e-12;
x(abs(x + pi) < tol) = pi;
end

function theta = wrap_to_nrosy_theta(theta, Nfold)
theta = wrap_to_pi(Nfold * theta) / Nfold;
end

function [Q, b1, b2] = local_minimal_Q_frames(fn)
nf = size(fn, 1);
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
        % pi rotation: choose any axis orthogonal to z; here x-axis.
        axis = [1; 0; 0];
        K = hat_local(axis);
        R = eye(3) + 2 * (K * K);
    else
        axis = cross(z, n);
        s = norm(axis);
        axis = axis / max(s, 1e-12);
        theta = atan2(s, c);
        K = hat_local(axis);
        % Rodrigues formula for minimal axis-angle rotation.
        R = eye(3) + sin(theta) * K + (1 - cos(theta)) * (K * K);
    end

    Q(:,:,f) = R;
    b1(f, :) = (R * [1; 0; 0])';
    b2(f, :) = (R * [0; 1; 0])';
end
end

function K = hat_local(v)
K = [0, -v(3), v(2); v(3), 0, -v(1); -v(2), v(1), 0];
end

function phi = logSO3_vee_local(R)
trv = 0.5 * (trace(R) - 1);
trv = min(1, max(-1, trv));
theta = acos(trv);
if theta < 1e-8
    phi = 0.5 * [R(3,2)-R(2,3); R(1,3)-R(3,1); R(2,1)-R(1,2)];
    return;
end
phi_hat = (theta / (2 * max(sin(theta), 1e-12))) * (R - R');
phi = [phi_hat(3,2); phi_hat(1,3); phi_hat(2,1)];
end
