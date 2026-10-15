function [dirsN, fname, mesh] = SolveNDirectionSO3Mix( ...
        X, T, mname, Visualize, FixedTriangle, FixedDirection, Nfold, mu, options)
%SOLVENDIRECTIONSO3MIX Solve an N-direction field with mixed energy.
% The optimized energy is
%
%   sum_(f,g) w_fg * sin(N * n_g' * logSO3(R_fg*R_f*R_g') / 2)^2
%   + mu * rho_fg * w_fg / 2
%       * ((u_f' * n_g)^2 + (u_g' * n_f)^2),
%
% where rho_fg = sin(eta_fg)^2 = 1 - (n_f' * n_g)^2.

if nargin < 5 || isempty(FixedTriangle)
    FixedTriangle = [];
end
if nargin < 6 || isempty(FixedDirection)
    FixedDirection = [];
end
if nargin < 7 || isempty(Nfold)
    Nfold = 4;
end
if nargin < 8 || isempty(mu)
    mu = 1;
end
if nargin < 9 || isempty(options)
    options = struct();
end
validateattributes(Nfold, {'numeric'}, ...
    {'scalar', 'integer', '>=', 1});
validateattributes(mu, {'numeric'}, {'scalar', 'real', 'finite', '>=', 0});

enableBranchSelection = getOption(options, 'EnableBranchSelection', true);

shouldSave = ~isempty(mname);
projectDir = fileparts(fileparts(mfilename('fullpath')));
resultsDir = fullfile(projectDir, 'Results');
if shouldSave
    if ~exist(resultsDir, 'dir')
        mkdir(resultsDir);
    end
    fname = fullfile(resultsDir, sprintf( ...
        'mesh_%s_N%d_ndirection_so3_mix_mu_%g.mat', mname, Nfold, mu));
else
    fname = '';
end

nf = size(T, 1);
nv = size(X, 1);
edges = unique(sort(reshape([T, T(:, [2 3 1])], [], 2), 2), 'rows');
ne = size(edges, 1);

edgeFace = sparse(T, T(:, [2 3 1]), repmat(1:nf, 1, 3), nv, nv);
edges2triangles = zeros(ne, 2);
for edgeIndex = 1:ne
    i = edges(edgeIndex, 1);
    j = edges(edgeIndex, 2);
    edges2triangles(edgeIndex, :) = [edgeFace(i, j), edgeFace(j, i)];
end
isBoundaryEdge = any(edges2triangles == 0, 2);
interiorEdges = find(~isBoundaryEdge);

edgeWeights = edge_weight(X, T, edges);
[~, ~, initialTheta, initData] = InitializeNFieldComplex( ...
    X, T, FixedTriangle, FixedDirection, Nfold);
faceNormals = initData.fn;
baseFrames = initData.Q;
connectionAngles = initData.beta;
transportRotations = initData.Rij;
fixedMask = initData.fixedMask;
fixed = find(fixedMask);
free = find(~fixedMask);

normalDot = sum( ...
    faceNormals(edges2triangles(interiorEdges, 1), :) .* ...
    faceNormals(edges2triangles(interiorEdges, 2), :), 2);
normalDot = max(-1, min(1, normalDot));
creaseWeights = zeros(ne, 1);
creaseWeights(interiorEdges) = max(0, 1 - normalDot.^2);

rawInitialTheta = initialTheta(:);
if enableBranchSelection
    [theta, preprocessing] = preprocessCreaseRepresentatives( ...
        rawInitialTheta, baseFrames, faceNormals, interiorEdges, ...
        edges2triangles, edgeWeights, creaseWeights, fixedMask, Nfold, mu);
else
    theta = rawInitialTheta;
    preprocessing = skippedPreprocessingInfo( ...
        rawInitialTheta, baseFrames, faceNormals, interiorEdges, ...
        edges2triangles, edgeWeights, creaseWeights, Nfold, mu);
end

mesh = struct();
mesh.edges = edges;
mesh.edges2triangles = edges2triangles;
mesh.isBoundaryEdge = isBoundaryEdge;
mesh.numInteriorEdges = numel(interiorEdges);
mesh.numBoundaryEdges = sum(isBoundaryEdge);
mesh.edgeWeights = edgeWeights;
mesh.creaseWeights = creaseWeights;
mesh.connectionAngles = connectionAngles;
mesh.transportRotations = transportRotations;
mesh.triangleBarycenters = ...
    (X(T(:,1), :) + X(T(:,2), :) + X(T(:,3), :)) / 3;
mesh.rawInitialTheta = rawInitialTheta;
mesh.initialTheta = theta;
mesh.initData = initData;
mesh.fixedMask = fixedMask;
mesh.rawInitDirsN = buildDirsN(baseFrames, rawInitialTheta, Nfold);
mesh.initDirsN = buildDirsN(baseFrames, theta, Nfold);
mesh.initializationPreprocessing = preprocessing;
mesh.enableBranchSelection = enableBranchSelection;

maxIterations = 300;
gradientTolerance = 1e-8;
stepTolerance = 1e-10;
armijoSlope = 1e-4;
maxLineSearch = 25;

[residual, jacobian, energyParts] = mixedResiduals( ...
    theta, baseFrames, faceNormals, interiorEdges, edges2triangles, ...
    edgeWeights, creaseWeights, transportRotations, Nfold, mu, true);
energy = residual' * residual;
energyHistory = zeros(maxIterations + 1, 1);
intrinsicHistory = zeros(maxIterations + 1, 1);
bendingHistory = zeros(maxIterations + 1, 1);
gradientHistory = zeros(maxIterations, 1);
stepHistory = zeros(maxIterations, 1);
energyHistory(1) = energy;
intrinsicHistory(1) = energyParts.intrinsic;
bendingHistory(1) = energyParts.bending;
acceptedIterations = 0;

for iteration = 1:maxIterations
    if isempty(free)
        break;
    end

    gradient = jacobian' * residual;
    gradientFree = gradient(free);
    gradientNorm = norm(gradientFree, inf);
    gradientHistory(iteration) = gradientNorm;
    if gradientNorm < gradientTolerance
        break;
    end

    hessianFree = jacobian(:, free)' * jacobian(:, free);
    damping = 1e-8;
    directionFree = [];
    for dampingAttempt = 1:12
        candidate = -(hessianFree + ...
            damping * speye(numel(free))) \ gradientFree;
        if all(isfinite(candidate)) && gradientFree' * candidate < 0
            directionFree = candidate;
            break;
        end
        damping = max(10 * damping, 1e-8);
    end
    if isempty(directionFree)
        directionFree = -gradientFree;
    end

    direction = zeros(nf, 1);
    direction(free) = directionFree;
    directionalDerivative = 2 * gradientFree' * directionFree;
    step = 1;
    accepted = false;
    for lineSearchIteration = 1:maxLineSearch
        thetaCandidate = theta + step * direction;
        [candidateResidual, ~, candidateParts] = mixedResiduals( ...
            thetaCandidate, baseFrames, faceNormals, interiorEdges, ...
            edges2triangles, edgeWeights, creaseWeights, ...
            transportRotations, Nfold, mu, false);
        candidateEnergy = candidateResidual' * candidateResidual;
        if isfinite(candidateEnergy) && candidateEnergy <= ...
                energy + armijoSlope * step * directionalDerivative
            accepted = true;
            break;
        end
        step = 0.5 * step;
    end

    if ~accepted
        break;
    end

    theta = thetaCandidate;
    energy = candidateEnergy;
    acceptedIterations = acceptedIterations + 1;
    energyHistory(acceptedIterations + 1) = energy;
    intrinsicHistory(acceptedIterations + 1) = candidateParts.intrinsic;
    bendingHistory(acceptedIterations + 1) = candidateParts.bending;
    stepHistory(acceptedIterations) = step * norm(directionFree);

    [residual, jacobian, ~] = mixedResiduals( ...
        theta, baseFrames, faceNormals, interiorEdges, edges2triangles, ...
        edgeWeights, creaseWeights, transportRotations, Nfold, mu, true);
    if stepHistory(acceptedIterations) < stepTolerance
        break;
    end
end

energyHistory = energyHistory(1:acceptedIterations + 1);
intrinsicHistory = intrinsicHistory(1:acceptedIterations + 1);
bendingHistory = bendingHistory(1:acceptedIterations + 1);
gradientHistory = gradientHistory(1:max(acceptedIterations, 1));
stepHistory = stepHistory(1:acceptedIterations);

frames = zeros(3 * nf, 3);
for faceIndex = 1:nf
    frame = baseFrames(:,:,faceIndex) * rotzLocal(theta(faceIndex));
    frames(3*faceIndex-2:3*faceIndex, :) = frame;
end
dir1 = reshape(frames(:,1), 3, [])';
dir2 = reshape(frames(:,2), 3, [])';
dirsN = buildDirsNFromDirections(dir1, dir2, Nfold);

mesh.energy = energyHistory;
mesh.intrinsicEnergy = intrinsicHistory;
mesh.bendingEnergy = bendingHistory;
mesh.initialEnergy = energyHistory(1);
mesh.finalEnergy = energyHistory(end);
mesh.gradientNorm = gradientHistory;
mesh.stepNorm = stepHistory;
mesh.iterations = acceptedIterations;
mesh.theta = theta;
mesh.frames = frames;
mesh.dir1 = dir1;
mesh.dir2 = dir2;
mesh.Nfold = Nfold;
mesh.mu = mu;
mesh.fixed = fixed;
mesh.free = free;
mesh.energyName = ['SO(3)-Log intrinsic N-smoothness + symmetric ', ...
    'crease-weighted bending regularization'];

if Visualize
    visualizeField(X, T, mesh.triangleBarycenters, dirsN, ...
        fixedMask, Nfold, mu);
end

if shouldSave
    toSave.frames = frames;
    toSave.dir1 = dir1;
    toSave.dir2 = dir2;
    toSave.dirsN = dirsN;
    toSave.fn = faceNormals;
    toSave.energy = energyHistory;
    toSave.intrinsicEnergy = intrinsicHistory;
    toSave.bendingEnergy = bendingHistory;
    toSave.Nfold = Nfold;
    toSave.mu = mu;
    toSave.theta = theta;
    toSave.rawInitialTheta = rawInitialTheta;
    toSave.initialTheta = mesh.initialTheta;
    toSave.initializationPreprocessing = preprocessing;
    toSave.enableBranchSelection = enableBranchSelection;
    toSave.FixedTriangle = FixedTriangle;
    toSave.FixedDirection = FixedDirection;
    toSave.fixed = fixed;
    toSave.free = free;
    toSave.X = X;
    toSave.T = T;
    toSave.mesh = mesh;
    save(fname, 'toSave');
    disp(['Saved result to: ', fname]);
end
end

function [theta, info] = preprocessCreaseRepresentatives( ...
        theta, baseFrames, faceNormals, interiorEdges, edges2triangles, ...
        edgeWeights, creaseWeights, fixedMask, Nfold, mu)
rotationStep = 2 * pi / Nfold;
normalAngleThreshold = pi / 4;
internalDihedralThreshold = normalAngleThreshold;
energyTolerance = 1e-14;

normalDot = sum( ...
    faceNormals(edges2triangles(interiorEdges, 1), :) .* ...
    faceNormals(edges2triangles(interiorEdges, 2), :), 2);
normalDot = max(-1, min(1, normalDot));
normalAngles = acos(normalDot);
selectedCreaseMask = normalAngles > normalAngleThreshold;
selectedCreaseEdges = interiorEdges(selectedCreaseMask);
[~, order] = sort(normalAngles(selectedCreaseMask), 'descend');
selectedCreaseEdges = selectedCreaseEdges(order);

faceInteriorEdges = cell(size(faceNormals, 1), 1);
for edgeIndex = interiorEdges'
    f = edges2triangles(edgeIndex, 1);
    g = edges2triangles(edgeIndex, 2);
    faceInteriorEdges{f}(end + 1) = edgeIndex;
    faceInteriorEdges{g}(end + 1) = edgeIndex;
end

initialBendingEnergy = totalBendingEnergy( ...
    theta, baseFrames, faceNormals, interiorEdges, edges2triangles, ...
    edgeWeights, creaseWeights, mu);
acceptedRotations = 0;

for edgeIndex = selectedCreaseEdges'
    adjacentFaces = edges2triangles(edgeIndex, :);
    for faceIndex = adjacentFaces
        if fixedMask(faceIndex)
            continue;
        end

        currentEnergy = incidentBendingEnergy( ...
            faceIndex, theta(faceIndex), baseFrames, faceNormals, ...
            faceInteriorEdges, edges2triangles, edgeWeights, ...
            creaseWeights, mu);
        candidateAngles = theta(faceIndex) + [-rotationStep, rotationStep];
        candidateEnergies = zeros(1, 2);
        for candidateIndex = 1:2
            candidateEnergies(candidateIndex) = incidentBendingEnergy( ...
                faceIndex, candidateAngles(candidateIndex), baseFrames, ...
                faceNormals, faceInteriorEdges, edges2triangles, ...
                edgeWeights, creaseWeights, mu);
        end

        [bestEnergy, bestIndex] = min(candidateEnergies);
        if bestEnergy < currentEnergy - ...
                energyTolerance * max(1, currentEnergy)
            theta(faceIndex) = candidateAngles(bestIndex);
            acceptedRotations = acceptedRotations + 1;
        end
    end
end

finalBendingEnergy = totalBendingEnergy( ...
    theta, baseFrames, faceNormals, interiorEdges, edges2triangles, ...
    edgeWeights, creaseWeights, mu);
assert(finalBendingEnergy <= initialBendingEnergy + ...
    1e-12 * max(1, initialBendingEnergy), ...
    'Crease preprocessing increased the bending energy.');

info = struct();
info.rotationStep = rotationStep;
info.internalDihedralThreshold = internalDihedralThreshold;
info.normalAngleThreshold = normalAngleThreshold;
info.numSelectedCreaseEdges = numel(selectedCreaseEdges);
info.acceptedRotations = acceptedRotations;
info.initialBendingEnergy = initialBendingEnergy;
info.finalBendingEnergy = finalBendingEnergy;
end

function value = getOption(options, fieldName, defaultValue)
if isstruct(options) && isfield(options, fieldName) && ...
        ~isempty(options.(fieldName))
    value = options.(fieldName);
else
    value = defaultValue;
end
end

function info = skippedPreprocessingInfo( ...
        theta, baseFrames, faceNormals, interiorEdges, edges2triangles, ...
        edgeWeights, creaseWeights, Nfold, mu)
energy = totalBendingEnergy( ...
    theta, baseFrames, faceNormals, interiorEdges, edges2triangles, ...
    edgeWeights, creaseWeights, mu);
info = struct();
info.rotationStep = 2 * pi / Nfold;
info.normalAngleThreshold = pi / 4;
info.internalDihedralThreshold = info.normalAngleThreshold;
info.numSelectedCreaseEdges = 0;
info.acceptedRotations = 0;
info.initialBendingEnergy = energy;
info.finalBendingEnergy = energy;
info.skipped = true;
end

function energy = incidentBendingEnergy( ...
        faceIndex, faceTheta, baseFrames, faceNormals, faceInteriorEdges, ...
        edges2triangles, edgeWeights, creaseWeights, mu)
frame = baseFrames(:,:,faceIndex) * rotzLocal(faceTheta);
direction = frame(:, 1);
energy = 0;

for edgeIndex = faceInteriorEdges{faceIndex}
    adjacentFaces = edges2triangles(edgeIndex, :);
    if adjacentFaces(1) == faceIndex
        otherFace = adjacentFaces(2);
    elseif adjacentFaces(2) == faceIndex
        otherFace = adjacentFaces(1);
    else
        continue;
    end

    coefficient = 0.5 * mu * max(creaseWeights(edgeIndex), 0) * ...
        max(edgeWeights(edgeIndex), 0);
    energy = energy + coefficient * ...
        (direction' * faceNormals(otherFace, :)')^2;
end
end

function energy = totalBendingEnergy( ...
        theta, baseFrames, faceNormals, interiorEdges, edges2triangles, ...
        edgeWeights, creaseWeights, mu)
energy = 0;
for edgeIndex = interiorEdges'
    f = edges2triangles(edgeIndex, 1);
    g = edges2triangles(edgeIndex, 2);
    frameF = baseFrames(:,:,f) * rotzLocal(theta(f));
    frameG = baseFrames(:,:,g) * rotzLocal(theta(g));
    directionF = frameF(:, 1);
    directionG = frameG(:, 1);
    coefficient = 0.5 * mu * max(creaseWeights(edgeIndex), 0) * ...
        max(edgeWeights(edgeIndex), 0);
    energy = energy + coefficient * ( ...
        (directionF' * faceNormals(g, :)')^2 + ...
        (directionG' * faceNormals(f, :)')^2);
end
end

function [residual, jacobian, parts] = mixedResiduals( ...
        theta, baseFrames, faceNormals, interiorEdges, edges2triangles, ...
        edgeWeights, creaseWeights, transportRotations, Nfold, mu, needJacobian)
nf = numel(theta);
numInteriorEdges = numel(interiorEdges);
residual = zeros(3 * numInteriorEdges, 1);
if needJacobian
    jacobianRows = zeros(4 * numInteriorEdges, 1);
    jacobianColumns = zeros(4 * numInteriorEdges, 1);
    jacobianValues = zeros(4 * numInteriorEdges, 1);
else
    jacobian = [];
end
intrinsicEnergy = 0;
bendingEnergy = 0;

for localEdgeIndex = 1:numInteriorEdges
    edgeIndex = interiorEdges(localEdgeIndex);
    f = edges2triangles(edgeIndex, 1);
    g = edges2triangles(edgeIndex, 2);
    weight = max(edgeWeights(edgeIndex), 0);
    rho = max(creaseWeights(edgeIndex), 0);

    frameF = baseFrames(:,:,f) * rotzLocal(theta(f));
    frameG = baseFrames(:,:,g) * rotzLocal(theta(g));
    directionF = frameF(:, 1);
    directionG = frameG(:, 1);
    directionDerivativeF = frameF(:, 2);
    directionDerivativeG = frameG(:, 2);

    transport = transportRotations(3*edgeIndex-2:3*edgeIndex, :);
    relativeRotation = transport * frameF * frameG';
    phi = logSO3Vee(relativeRotation);
    normalG = faceNormals(g, :)';
    intrinsicAngle = normalG' * phi;
    intrinsicScale = sqrt(weight);
    intrinsicPhase = 0.5 * Nfold * intrinsicAngle;
    bendingScale = sqrt(0.5 * mu * rho * weight);

    row = 3 * localEdgeIndex - 2;
    residual(row) = intrinsicScale * sin(intrinsicPhase);
    residual(row + 1) = bendingScale * (directionF' * faceNormals(g, :)');
    residual(row + 2) = bendingScale * (directionG' * faceNormals(f, :)');

    intrinsicEnergy = intrinsicEnergy + residual(row)^2;
    bendingEnergy = bendingEnergy + ...
        residual(row + 1)^2 + residual(row + 2)^2;

    if ~needJacobian
        continue;
    end

    intrinsicDerivative = intrinsicScale * ...
        0.5 * Nfold * cos(intrinsicPhase) * ...
        (normalG' * rightJacobianInverseSO3(phi) * normalG);
    entry = 4 * localEdgeIndex - 3;
    jacobianRows(entry:entry+3) = [row; row; row + 1; row + 2];
    jacobianColumns(entry:entry+3) = [f; g; f; g];
    jacobianValues(entry:entry+3) = [ ...
        intrinsicDerivative; ...
        -intrinsicDerivative; ...
        bendingScale * (directionDerivativeF' * faceNormals(g, :)'); ...
        bendingScale * (directionDerivativeG' * faceNormals(f, :)')];
end

if needJacobian
    jacobian = sparse(jacobianRows, jacobianColumns, jacobianValues, ...
        3 * numInteriorEdges, nf);
end

parts = struct();
parts.intrinsic = intrinsicEnergy;
parts.bending = bendingEnergy;
parts.total = intrinsicEnergy + bendingEnergy;
end

function phi = logSO3Vee(R)
cosAngle = 0.5 * (trace(R) - 1);
cosAngle = min(1, max(-1, cosAngle));
angle = acos(cosAngle);
if angle < 1e-8
    phi = 0.5 * [ ...
        R(3,2) - R(2,3); ...
        R(1,3) - R(3,1); ...
        R(2,1) - R(1,2)];
    return;
end
phiHat = (angle / (2 * max(sin(angle), 1e-12))) * (R - R');
phi = [phiHat(3,2); phiHat(1,3); phiHat(2,1)];
end

function inverseJacobian = rightJacobianInverseSO3(phi)
angle = norm(phi);
phiHat = [ ...
    0, -phi(3), phi(2); ...
    phi(3), 0, -phi(1); ...
    -phi(2), phi(1), 0];
if angle < 1e-6
    inverseJacobian = eye(3) + ...
        0.5 * phiHat + (1 / 12) * (phiHat * phiHat);
else
    coefficient = 1 / angle^2 - ...
        (1 + cos(angle)) / (2 * angle * sin(angle));
    inverseJacobian = eye(3) + ...
        0.5 * phiHat + coefficient * (phiHat * phiHat);
end
end

function R = rotzLocal(angle)
R = [cos(angle), -sin(angle), 0; ...
     sin(angle),  cos(angle), 0; ...
              0,           0, 1];
end

function dirsN = buildDirsN(baseFrames, theta, Nfold)
nf = numel(theta);
dirsN = zeros(3, Nfold, nf);
for faceIndex = 1:nf
    frame = baseFrames(:,:,faceIndex) * rotzLocal(theta(faceIndex));
    for directionIndex = 1:Nfold
        angle = 2 * pi * (directionIndex - 1) / Nfold;
        dirsN(:, directionIndex, faceIndex) = ...
            cos(angle) * frame(:, 1) + sin(angle) * frame(:, 2);
    end
end
end

function dirsN = buildDirsNFromDirections(dir1, dir2, Nfold)
nf = size(dir1, 1);
dirsN = zeros(3, Nfold, nf);
for faceIndex = 1:nf
    for directionIndex = 1:Nfold
        angle = 2 * pi * (directionIndex - 1) / Nfold;
        dirsN(:, directionIndex, faceIndex) = ...
            (cos(angle) * dir1(faceIndex, :) + ...
            sin(angle) * dir2(faceIndex, :))';
    end
end
end

function visualizeField(X, T, centers, dirsN, fixedMask, Nfold, mu)
nf = size(T, 1);
figure('Name', 'Mixed extrinsic N-direction field');
hold on;
axis equal;
rotate3d on;
patch('Faces', T, 'Vertices', X, ...
    'FaceColor', [0.82 0.82 0.82], 'EdgeColor', [0.45 0.45 0.45]);
title(sprintf('%d-Direction Field (mixed, \\mu = %g)', Nfold, mu));

origins = repelem(centers, Nfold, 1);
directions = zeros(nf * Nfold, 3);
fixedDirections = false(nf * Nfold, 1);
cursor = 1;
for faceIndex = 1:nf
    for directionIndex = 1:Nfold
        directions(cursor, :) = dirsN(:, directionIndex, faceIndex)';
        fixedDirections(cursor) = fixedMask(faceIndex);
        cursor = cursor + 1;
    end
end

quiver3(origins(~fixedDirections,1), origins(~fixedDirections,2), ...
    origins(~fixedDirections,3), directions(~fixedDirections,1), ...
    directions(~fixedDirections,2), directions(~fixedDirections,3), ...
    0.5, 'b');
if any(fixedDirections)
    quiver3(origins(fixedDirections,1), origins(fixedDirections,2), ...
        origins(fixedDirections,3), directions(fixedDirections,1), ...
        directions(fixedDirections,2), directions(fixedDirections,3), ...
        0.5, 'r', 'LineWidth', 1.5);
end
end
