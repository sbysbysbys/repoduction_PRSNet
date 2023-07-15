function [] = generateSurfacePlot()
    %generateSurface('..//results//compare//test_data_11.mat');
    generateSurface('..//results//comparel//compare//test_data_14.mat');
end

function generateSurface(tsdfFile)
    load(tsdfFile);

    varInfo = whos('-file', tsdfFile);
    
    figure;
    
    vertices = eval(varInfo(strcmp({varInfo.name}, 'vertices')).name);

    if ~exist('vertices', 'var')
        error('Variable "vertices" is not initialized.');
    end

    faces = eval(varInfo(strcmp({varInfo.name}, 'faces')).name);

    trisurf(faces, vertices(:, 1), vertices(:, 2), vertices(:, 3), 'FaceColor', 'interp', 'EdgeColor', 'none');
    axis([-1 1 -1 1 -1 1]);
    axis equal;
    
    % plane
    if ismember("plane1", {varInfo.name})
        hold on;
        plane1 = eval(varInfo(strcmp({varInfo.name}, 'plane1')).name);
        a = plane1(1);
        b = plane1(2);
        c = plane1(3);
        d = plane1(4);
        
        [x, y] = meshgrid(min(vertices(:, 1)):0.1:max(vertices(:, 1)), min(vertices(:, 2)):0.1:max(vertices(:, 2)));
        z = -(a * x + b * y + d) / c;
        
        surf(x, y, z, 'FaceColor', 'r', 'FaceAlpha', 0.5);
        disp('Value of parameter "plane1":');
        disp(plane1);
    end

    if ismember("plane2", {varInfo.name})
        hold on;
        plane2 = eval(varInfo(strcmp({varInfo.name}, 'plane2')).name);

        a = plane2(1);
        b = plane2(2);
        c = plane2(3);
        d = plane2(4);
        
        [x, y] = meshgrid(min(vertices(:, 1)):0.1:max(vertices(:, 1)), min(vertices(:, 2)):0.1:max(vertices(:, 2)));
        z = -(a * x + b * y + d) / c;
        
        surf(x, y, z, 'FaceColor', 'r', 'FaceAlpha', 0.5);

        disp('Value of parameter "plane2":');
        disp(plane2);
    end

    if ismember("plane3", {varInfo.name})
        hold on;
        plane3 = eval(varInfo(strcmp({varInfo.name}, 'plane3')).name);

        a = plane3(1);
        b = plane3(2);
        c = plane3(3);
        d = plane3(4);
        
        [x, y] = meshgrid(min(vertices(:, 1)):0.1:max(vertices(:, 1)), min(vertices(:, 2)):0.1:max(vertices(:, 2)));
        z = -(a * x + b * y + d) / c;
        
        surf(x, y, z, 'FaceColor', 'r', 'FaceAlpha', 0.5);

        disp('Value of parameter "plane3":');
        disp(plane3);
    end

    %quat

    if ismember("quat1", {varInfo.name})
        hold on;
        quat1 = eval(varInfo(strcmp({varInfo.name}, 'quat1')).name);
        
        ui0 = quat1(1);
        ui1 = quat1(2);
        ui2 = quat1(3);
        ui3 = quat1(4);
        
        line([0 ui1], [0 ui2], [0 ui3], 'Color', 'b', 'LineWidth', 2);

        disp('Value of parameter "quat1":');
        disp(quat1);
    end

    if ismember("quat2", {varInfo.name})
        hold on;
        quat2 = eval(varInfo(strcmp({varInfo.name}, 'quat2')).name);
        
        ui0 = quat2(1);
        ui1 = quat2(2);
        ui2 = quat2(3);
        ui3 = quat2(4);
        
        line([0 ui1], [0 ui2], [0 ui3], 'Color', 'b', 'LineWidth', 2);

        disp('Value of parameter "quat2":');
        disp(quat2);
    end

    if ismember("quat3", {varInfo.name})
        hold on;
        quat3 = eval(varInfo(strcmp({varInfo.name}, 'quat3')).name);
        
        ui0 = quat3(1);
        ui1 = quat3(2);
        ui2 = quat3(3);
        ui3 = quat3(4);
        
        line([0 ui1], [0 ui2], [0 ui3], 'Color', 'b', 'LineWidth', 2);
        
        disp('Value of parameter "quat3":');
        disp(quat3);
    end
    
    axis([-1 1 -1 1 -1 1]);
    title('Surface Plot');
    xlabel('X');
    ylabel('Y');
    zlabel('Z');
end




