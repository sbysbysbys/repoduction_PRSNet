function [] = visPointsPlaneAndRot()

    point1 = [0.0610, 0.0449, 0.1178];
    point2 = [0.1225, 0.0174, -0.0685];

    quat = [-0.745, 0.3975, 0.0606, 0.1240];

    figure;
    hold on;

    % 绘制平面
    % visPlane(0.1144, 0.0205, 0.0096, 0.0020);

    % 绘制点
    visPoint(point1);
    visPoint(point2);

    % 绘制旋转轴
    visRotationAxis(quat);

    axis([-1 1 -1 1 -1 1]);
    axis equal;

    xlabel('X');
    ylabel('Y');
    zlabel('Z');
    title('Points, Plane, and Rotation Axis');
    axis equal;
    grid on;
    view(3);
end

function [] = visPlane(a, b, c, d)
    [X, Y] = meshgrid(-5:0.2:5);
    Z = (-d - a*X - b*Y) / c;

    surf(X, Y, Z, 'FaceAlpha', 0.5);
end

function [] = visPoint(point)
    plot3(point(1), point(2), point(3), 'ro', 'MarkerSize', 10, 'LineWidth', 2);
end

function [] = visRotationAxis(quat)
    R = quat2rotm(quat);
    [V, D] = eig(R);
    [~, ind] = max(abs(diag(D)));
    axis = V(:, ind);
    plot3([0, axis(1)], [0, axis(2)], [0, axis(3)], 'b-', 'LineWidth', 2);
end


