function [fixface, fixdir, fixedge] = GetFeatureAndBoundaryFaces(x, t, angle_threshold_deg)
% GetFeatureAndBoundaryFaces - 提取网格的边界边和特征边，以及它们相邻的面信息
%
% 输入:
%   x                  - nv×3 顶点位置
%   t                  - nf×3 网格面的顶点列表
%   angle_threshold_deg - 特征边二面角阈值（度数），超过该阈值则确定为特征边
%
% 输出:
%   fixface  - n×1 特征边或边界边相邻的面序号（指向t中的第几行）
%              如果一个面与两条或更多特征边/边界边相邻，则不计入
%   fixdir   - n×3 与fixface行行对应的单位方向向量
%              指向与该面对应的特征边或边界边的平行单位方向向量
%   fixedge  - n×2 与fixface行行对应的特征边或边界边顶点序号

    %% Step 1: 构建边的邻接矩阵
    nf = size(t, 1);
    nv = size(x, 1);

    % 面法向量
    v1 = x(t(:,2),:) - x(t(:,1),:);
    v2 = x(t(:,3),:) - x(t(:,1),:);
    fn = cross(v1, v2, 2);
    fn = fn ./ max(1e-12, sqrt(sum(fn.^2, 2))); % 单位面法向量

    % 邻接关系: adjmat(i,j) = 包含有向边 (i,j) 的面序号
    adjmat = sparse(t, t(:, [2 3 1]), repmat(1:nf, 1, 3), nv, nv);
    mask = adjmat & adjmat';
    tmp = adjmat .* mask;
    adjmat(mask) = 0;
    adjmat = tmp + adjmat + adjmat';

    %% Step 2: 提取所有边（无向）
    e = unique(sort(reshape([t, t(:, [2 3 1])], [], 2), 2), 'rows');
    ne = size(e, 1);

    %% Step 3: 提取特征边
    % 对每条边计算二面角，如果超过阈值则为特征边
    sharpEdges = [];
    angle_threshold = deg2rad(angle_threshold_deg);

    for ind = 1:ne
        i = e(ind, 1);
        j = e(ind, 2);
        
        a = adjmat(i, j);
        b = adjmat(j, i);

        % 非流形边跳过
        if a == 0 || b == 0
            continue;
        end

        % 计算二面角
        n1 = fn(a, :);
        n2 = fn(b, :);
        dihedral = acos(dot(n1, n2));

        if dihedral > angle_threshold
            sharpEdges(end+1, :) = [i, j];
        end
    end

    %% Step 4: 提取边界边
    % 使用 findBoundary 的逻辑获取边界顶点环
    [B, H] = findBoundary(x, t);

    Be = [];
    if ~isempty(B)
        B1 = [B(2:end)'; B(1)];
        Be = [B', B1];
    end
    if ~isempty(H)
        for ind = 1:numel(H)
            h = H{ind};
            h1 = [h(2:end)'; h(1)];
            he = [h1, h'];
            Be = [Be; he];
        end
    end

    %% Step 5: 合并特征边和边界边
    sharpEdges = unique(sort(sharpEdges, 2), 'rows');
    Be = unique(sort(Be, 2), 'rows');
    if isempty(Be) && ~isempty(sharpEdges)
        fe = sharpEdges;
    elseif ~isempty(Be) && isempty(sharpEdges)
        fe = Be;
    elseif ~isempty(Be) && ~isempty(sharpEdges)
        fe = union(sharpEdges, Be, 'rows');
    else
        error('Lack of constraints: no boundary or feature edges found');
    end

    %% Step 6: 提取每个特征边/边界边相邻的面，并计算平行单位方向向量
    % 先统计每个面与多少条特征边/边界边相邻
    face_count = zeros(nf, 1);
    face_edge_dir = cell(nf, 1); % 每个面存储对应的边方向向量
    face_edge_vertices = cell(nf, 1);

    for ind = 1:size(fe, 1)
        i = fe(ind, 1);
        j = fe(ind, 2);
        
        a = adjmat(i, j);
        b = adjmat(j, i);
        
        % 计算边的方向（从i到j的单位向量）
        tmp = x(j, :) - x(i, :);
        edgeLength = norm(tmp);
        if edgeLength < 1e-12
            continue;
        end
        tmp = tmp ./ edgeLength;
        
        % 面a
        if a > 0
            face_count(a) = face_count(a) + 1;
            face_edge_dir{a}(end+1, :) = tmp;
            face_edge_vertices{a}(end+1, :) = [i, j];
        end
        
        % 面b；边界边在对称邻接中可能有a==b，同一面只记录一次
        if b > 0 && b ~= a
            face_count(b) = face_count(b) + 1;
            face_edge_dir{b}(end+1, :) = tmp;
            face_edge_vertices{b}(end+1, :) = [i, j];
        end
    end

    %% Step 7: 只保留恰好与一条特征边/边界边相邻的面
    idx_one_edge = find(face_count == 1);
    
    fixface = idx_one_edge;
    fixdir = zeros(length(idx_one_edge), 3);
    fixedge = zeros(length(idx_one_edge), 2);
    
    for k = 1:length(idx_one_edge)
        fid = idx_one_edge(k);
        edgeDirection = face_edge_dir{fid}(1, :);
        edgeDirection = edgeDirection - ...
            dot(edgeDirection, fn(fid, :)) * fn(fid, :);
        fixdir(k, :) = edgeDirection ./ max(norm(edgeDirection), 1e-12);
        fixedge(k, :) = face_edge_vertices{fid}(1, :);
    end

end
