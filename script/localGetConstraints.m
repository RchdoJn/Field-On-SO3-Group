function [fixface, fixdir] = localGetConstraints( ...
        X, T, faceNormals, featureAngleDeg)
try
    [fixface, fixdir, fixedge] = GetFeatureAndBoundaryFaces( ...
        X, T, featureAngleDeg);
    for constraintIndex = 1:numel(fixface)
        edge = fixedge(constraintIndex, :);
        edgeDirection = X(edge(2), :) - X(edge(1), :);
        fixdir(constraintIndex, :) = edgeDirection ./ ...
            max(norm(edgeDirection), 1e-12);
    end
catch exception
    if ~contains(exception.message, 'Lack of constraints')
        rethrow(exception);
    end
    fixface = [];
    fixdir = [];
end

if isempty(fixface)
    fixface = 1;
    [~, ~, ~, secondPrincipalDirections] = PrincipalDirectFace(X, T);
    fixdir = secondPrincipalDirections(fixface, :);
    if ~all(isfinite(fixdir)) || norm(fixdir) < 1e-12
        fixdir = localAnyTangent(faceNormals(fixface, :));
    else
        fixdir = fixdir / norm(fixdir);
    end
    fprintf(['No boundary/feature constraints found. ', ...
        'Use the second principal-curvature direction on face 1 as fallback.\n']);
end
end

function tangent = localAnyTangent(normal)
axis = [1, 0, 0];
if abs(dot(axis, normal)) > 0.9
    axis = [0, 1, 0];
end
tangent = axis - dot(axis, normal) * normal;
tangent = tangent / max(norm(tangent), 1e-12);
end
