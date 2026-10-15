% test_mix.m
% Solve and validate an N-direction field with the mixed energy

clear; close all; clc;

thisDir = fileparts(mfilename('fullpath'));
addpath(thisDir);
addpath(fullfile(thisDir, 'script'));

meshFile = fullfile(thisDir, 'Meshes', 'fandisk.obj');
Nfold = 4;
mu = 100;
featureAngleDeg = 45;
ShouldVisualize = true;

[X, T] = readObj(meshFile);
[~, meshName, ~] = fileparts(meshFile);

v1 = X(T(:,2), :) - X(T(:,1), :);
v2 = X(T(:,3), :) - X(T(:,1), :);
faceNormals = cross(v1, v2, 2);
faceNormals = faceNormals ./ max(1e-12, sqrt(sum(faceNormals.^2, 2)));

[fixface, fixdir] = localGetConstraints( ...
    X, T, faceNormals, featureAngleDeg);

fprintf('Running SolveNDirectionSO3Mix on %s...\n', meshName);
[dirsN, saveFile, mesh] = SolveNDirectionSO3Mix( ...
    X, T, meshName, false, fixface, fixdir, Nfold, mu);

if ShouldVisualize
    assert(isequal(size(mesh.initDirsN), size(dirsN)), ...
        'Initialization and optimized fields have different dimensions.');
    assert(isequal(size(mesh.rawInitDirsN), size(dirsN)), ...
        'Raw initialization and optimized fields have different dimensions.');
    visualizeInitializationComparison( ...
        X, T, mesh.triangleBarycenters, mesh.rawInitDirsN, ...
        mesh.initDirsN, dirsN, fixface, Nfold, mu);
end

assert(isequal(size(dirsN), [3, Nfold, size(T, 1)]), ...
    'Unexpected dirsN dimensions.');
assert(all(isfinite(dirsN(:))), 'dirsN contains non-finite values.');
energyTolerance = 1e-9 * max(1, mesh.initialEnergy);
assert(all(diff(mesh.energy) <= energyTolerance), ...
    'Accepted optimization steps must not increase the mixed energy.');
assert(mesh.finalEnergy <= mesh.initialEnergy + energyTolerance, ...
    'Final mixed energy is larger than the initial energy.');
assert(max(abs(mesh.energy - ...
    (mesh.intrinsicEnergy + mesh.bendingEnergy))) < energyTolerance, ...
    'Stored energy components do not sum to the total energy.');

dir1 = reshape(dirsN(:, 1, :), 3, [])';
lengthError = max(abs(sqrt(sum(dir1.^2, 2)) - 1));
tangentError = max(abs(sum(dir1 .* faceNormals, 2)));
assert(lengthError < 1e-8, 'Output directions are not unit length.');
assert(tangentError < 1e-8, 'Output directions are not tangent to the mesh.');

for constraintIndex = 1:numel(fixface)
    faceIndex = fixface(constraintIndex);
    target = fixdir(constraintIndex, :) - ...
        dot(fixdir(constraintIndex, :), faceNormals(faceIndex, :)) * ...
        faceNormals(faceIndex, :);
    target = target / max(norm(target), 1e-12);
    alignment = max(arrayfun(@(directionIndex) ...
        dot(dirsN(:, directionIndex, faceIndex), target), 1:Nfold));
    assert(alignment > 1 - 1e-8, ...
        'A fixed direction was not preserved modulo N-fold symmetry.');
end

fprintf('Mixed-energy test passed.\n');
fprintf('  faces: %d\n', size(T, 1));
fprintf('  interior edges: %d\n', mesh.numInteriorEdges);
fprintf('  accepted iterations: %d\n', mesh.iterations);
fprintf('  initial energy: %.12g\n', mesh.initialEnergy);
fprintf('  final energy: %.12g\n', mesh.finalEnergy);
fprintf('  final intrinsic energy: %.12g\n', mesh.intrinsicEnergy(end));
fprintf('  final bending energy: %.12g\n', mesh.bendingEnergy(end));
fprintf('  selected high-normal-angle edges: %d\n', ...
    mesh.initializationPreprocessing.numSelectedCreaseEdges);
fprintf('  minimum selected normal angle: %.6g deg\n', ...
    rad2deg(mesh.initializationPreprocessing.normalAngleThreshold));
fprintf('  maximum selected internal dihedral: %.6g deg\n', ...
    rad2deg(mesh.initializationPreprocessing.internalDihedralThreshold));
fprintf('  representative rotation step: %.6g deg\n', ...
    rad2deg(mesh.initializationPreprocessing.rotationStep));
fprintf('  accepted initialization rotations: %d\n', ...
    mesh.initializationPreprocessing.acceptedRotations);
fprintf('  preprocessing bending energy: %.12g -> %.12g\n', ...
    mesh.initializationPreprocessing.initialBendingEnergy, ...
    mesh.initializationPreprocessing.finalBendingEnergy);
fprintf('  maximum unit-length error: %.3e\n', lengthError);
fprintf('  maximum tangency error: %.3e\n', tangentError);
fprintf('  saved result: %s\n', saveFile);

assignin('base', 'dirsN_mix', dirsN);
assignin('base', 'mesh_mix', mesh);
assignin('base', 'fixface', fixface);
assignin('base', 'fixdir', fixdir);

function visualizeInitializationComparison( ...
        X, T, centers, rawInitDirsN, initDirsN, finalDirsN, ...
        fixface, Nfold, mu)
fixedMask = false(size(T, 1), 1);
fixedMask(fixface) = true;

rawFigure = figure('Name', 'Raw mixed-field initialization');
rawAxes = axes(rawFigure);
drawField(rawAxes, X, T, centers, rawInitDirsN, fixedMask, Nfold);
title(rawAxes, sprintf('Raw initialization (%d-direction)', Nfold));

preprocessedFigure = figure('Name', 'Crease-preprocessed initialization');
preprocessedAxes = axes(preprocessedFigure);
drawField(preprocessedAxes, X, T, centers, initDirsN, fixedMask, Nfold);
title(preprocessedAxes, 'Crease-preprocessed initialization');

finalFigure = figure('Name', 'Optimized mixed field');
finalAxes = axes(finalFigure);
drawField(finalAxes, X, T, centers, finalDirsN, fixedMask, Nfold);
title(finalAxes, sprintf('Optimized result (\\mu = %g)', mu));

cameraLink = linkprop([rawAxes, preprocessedAxes, finalAxes], ...
    {'CameraPosition', 'CameraTarget', 'CameraUpVector', 'CameraViewAngle'});
setappdata(rawFigure, 'CameraLink', cameraLink);
setappdata(preprocessedFigure, 'CameraLink', cameraLink);
setappdata(finalFigure, 'CameraLink', cameraLink);
rotate3d(rawFigure, 'on');
rotate3d(preprocessedFigure, 'on');
rotate3d(finalFigure, 'on');
end

function drawField(axesHandle, X, T, centers, dirsN, fixedMask, Nfold)
hold(axesHandle, 'on');
axis(axesHandle, 'equal');
axis(axesHandle, 'off');
view(axesHandle, 3);
patch(axesHandle, 'Faces', T, 'Vertices', X, ...
    'FaceColor', [0.82 0.82 0.82], 'EdgeColor', [0.45 0.45 0.45]);

nf = size(T, 1);
origins = repelem(centers, Nfold, 1);
directions = zeros(nf * Nfold, 3);
fixedDirections = repelem(fixedMask, Nfold, 1);
cursor = 1;
for faceIndex = 1:nf
    for directionIndex = 1:Nfold
        directions(cursor, :) = dirsN(:, directionIndex, faceIndex)';
        cursor = cursor + 1;
    end
end

quiver3(axesHandle, origins(~fixedDirections,1), ...
    origins(~fixedDirections,2), origins(~fixedDirections,3), ...
    directions(~fixedDirections,1), directions(~fixedDirections,2), ...
    directions(~fixedDirections,3), 0.5, 'b');
if any(fixedDirections)
    quiver3(axesHandle, origins(fixedDirections,1), ...
        origins(fixedDirections,2), origins(fixedDirections,3), ...
        directions(fixedDirections,1), directions(fixedDirections,2), ...
        directions(fixedDirections,3), 0.5, 'r', 'LineWidth', 1.5);
end
end
