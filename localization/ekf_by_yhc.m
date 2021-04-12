clear all 
close all 
% last_mean  -------- [x,y,theta]
% 
% 
n_robots = 1;
robot_num = 1;
delta_t = 0.02
% load and resample raw data from UTIAS data set
% The robot's position groundtruth, odometry, and measurements 
% are stored in Robots
% 加载数据
[Barcodes, Landmark_Groundtruth, Robots] = loadMRCLAMdataSet(n_robots);
[Robots, timesteps] = sampleMRCLAMdataSet(Robots, delta_t);
% 这个Est是用来存储ekf的输出值，最后与groundtruth进行对比
Robots{robot_num}.Est = zeros(size(Robots{robot_num}.G,1), 4);
start = 634;
current_t = Robots{robot_num}.G(start, 1); % set start time

% set starting pose mean to pose groundtruth at start time
last_mean = [Robots{robot_num}.G(start,2);
            Robots{robot_num}.G(start,3);
            Robots{robot_num}.G(start,4)];
last_cov = [0.01 0.01 0.01;
           0.01 0.01 0.01;
           0.01 0.01 0.01];
% 这个是将barcode和landmark标签进行映射
codeDict = containers.Map(Barcodes(:,2),Barcodes(:,1));
measurementIndex = 1;
% 这个while是为了定位与当前时间帧对应的measurement
while (Robots{robot_num}.M(measurementIndex, 1) < current_t - .05)
        measurementIndex = measurementIndex + 1;
end
% 开始预测，从start对应的时间帧开始
for i = start : size(Robots{robot_num}.G, 1)
    current_t = Robots{robot_num}.G(i, 1);
    u = [Robots{robot_num}.O(i, 2); Robots{robot_num}.O(i, 3)];
    [Mean, Cov, measurementIndex] = EKF_with_Known(last_mean, last_cov, ...
                                      u, delta_t, Robots, current_t, codeDict,...
                                      measurementIndex, Landmark_Groundtruth);
    Mean(3) = conBear(Mean(3));
    Robots{robot_num}.Est(i,:) = [current_t Mean(1) Mean(2) Mean(3)];                
    last_mean = Mean;
    last_cov = Cov;
end
% 这个是显示ekf的结果
animateMRCLAMdataSet(Robots, Landmark_Groundtruth, timesteps, delta_t);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PREDICTION
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Mean, Cov, measurementIndex] = EKF_with_Known(last_mean, last_cov, ...
                                      u, delta_t, Robots, current_t, codeDict,...
                                      measurementIndex, Landmark_Groundtruth)
%input
% last_mean - dim 3x1
% Robots - 里面存放着ground truth, odometry, and measurement
% current - 当前时间帧，可以用来定位measurement时刻
% codedict  - map
% 

    theta = last_mean(3);
    alpha = [.2 .03 .09 .08 0 0]; % robot-dependent motion noise parameters
    v_t = u(1); % linear velocity
    omega_t = u(2); % angular velocity
    G_t = [1, 0, -v_t * delta_t * sin(theta + omega_t*delta_t);
          0, 1, v_t * delta_t * cos(theta + omega_t * delta_t);
          0, 0, 1];

    temp_angle = theta + omega_t * delta_t;  
    V_t = [cos(temp_angle), -0.5 * sin(temp_angle);
            sin(temp_angle), 0.5 * cos(temp_angle);
            0, 1];
    M_t = [alpha(1) * v_t^2 + alpha(2) * omega_t^2, 0;
            0, alpha(3) * v_t^2 + alpha(4) * omega_t^2];
%    prediction
    est_mean = last_mean + [v_t * delta_t * cos(temp_angle);
        v_t * delta_t * sin(temp_angle); 
        omega_t * delta_t];
    est_cov = G_t * last_cov * G_t.' + V_t * M_t * V_t.'; 
    robot_num = 1;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % update
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    [z, measurementIndex] = getObservations(Robots, robot_num, current_t, measurementIndex, codeDict);
    if z(3,1) > 1
        [est_mean, est_cov] = ekf_update(est_mean, est_cov, z, Landmark_Groundtruth);        
    end
    Mean = est_mean;
    Cov = est_cov;
    
end

function [Mean, Cov] = ekf_update(est_mean, est_cov, z, Landmark_Groundtruth)
% input 
% z ---  dim  3 * k  k indicates the number of seen landmarks
    sigma_range = 2;
    sigma_bearing = 3;
    sigma_id = 1;
    Q_t = [sigma_range^2 0 0;
           0 sigma_bearing^2 0;
           0 0 sigma_id^2];
    for k = 1 :size(z,2)
        % extract the landmark id
        ID = z(3,k);
        % extract the position of landmark with IDs
        m_t_k = Landmark_Groundtruth(ID, 2:3);
%         m_t_k = Landmark_Groundtruth(ID, 2:3);
        x_diff = m_t_k(1) - est_mean(1);
        y_diff = m_t_k(2) - est_mean(2);
        q = x_diff^2 + y_diff^2;
        bear_diff = conBear(atan2(y_diff, x_diff) - est_mean(3));
        z_bar = [sqrt(q); 
                 bear_diff; 
                 ID];
        
        H = [-(x_diff / sqrt(q)), -(y_diff / sqrt(q)), 0;
                 y_diff / q, -(x_diff / q), -1;
                 0, 0, 0];
        S = H * est_cov * H.' + Q_t;
%         K = est_cov * H.'*inv(S);
        K = (est_cov * H.')/S;
        est_mean = est_mean + K * (z(:, k) - z_bar );
        est_cov = (eye(3) - K*H) * est_cov;
    end
        Mean = est_mean;
        Cov = est_cov;

end