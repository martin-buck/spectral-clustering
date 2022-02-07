% -------------------------------------------------------------------------
% This is a demo code that generates concentric circles
% -------------------------------------------------------------------------
angle = 0:0.1:2*pi; 
radius = [0.5;2];
X_coordinate = [];
Y_coordinate = [];
% Create the concetric circles
for i = 1:2
    % Add some noise
    x = radius(i)*cos(angle);
    y = radius(i)*sin(angle);
    x = x';
    y = y';
    X_coordinate = [X_coordinate ;x];
    Y_coordinate = [Y_coordinate; y];
end
% Data matrix
X = [X_coordinate Y_coordinate];