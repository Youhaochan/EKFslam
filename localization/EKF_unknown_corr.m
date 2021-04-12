% reference link http://andrewjkramer.net/intro-to-the-ekf-step-1/
% 上面的代码还有对应的github,里面有数据和全套代码
addpath ../common;
close all 
clear all
deltaT = .02;
alphas = [.2 .03 .09 .08 0 0]; % motion model noise parameters

% measurement model noise parameters
sigma_range = .43;
sigma_bearing = .6;

Q_t = [sigma_range^2 0;
       0 sigma_bearing^2];

n_robots = 1;
robot_num = 1;
n_landmarks = 15;
% 加载数据，下面两行要在一起使用
[Barcodes, Landmark_Groundtruth, Robots] = loadMRCLAMdataSet(n_robots);
[Robots, timesteps] = sampleMRCLAMdataSet(Robots, deltaT);

% add pose estimate matrix to Robots
% 下面这行是用来存储ekf的输出值，用来最后与groundtruth进行对比
Robots{robot_num}.Est = zeros(size(Robots{robot_num}.G,1), 4);

start = 1; 
% 获取机器人的位姿真实值
t = Robots{robot_num}.G(start, 1); 
% need mean and covariance for the initial pose estimate
% 将start对应的位姿真实值作为机器人的初始位姿
poseMean = [Robots{robot_num}.G(start,2);
            Robots{robot_num}.G(start,3);
            Robots{robot_num}.G(start,4)];
poseCov = 0.01*eye(3);
%   测量值索引，使用来找到测量值位置的    
measurementIndex = 1;

% set up map between barcodes and landmark IDs
% 这个是将barcode与landmarkID映射成一个字典key(barcode)-value(landmark ID)
codeDict = containers.Map(Barcodes(:,2),Barcodes(:,1));

% 首先找到当前时间帧t对应的测量值数据对应的索引measurementIndex
while (Robots{robot_num}.M(measurementIndex, 1) < t - .05)
        measurementIndex = measurementIndex + 1;
end
IDS_comp = [];
% loop through all odometry and measurement samples
% updating the robot's pose estimate with each step
% reference table 7.3 in Probabilistic Robotics
% 从start对应的时间帧开始运算数据
for i = start:size(Robots{robot_num}.G, 1)
    theta = poseMean(3, 1);
    % update time
    t = Robots{robot_num}.G(i, 1);
    % update movement vector
    % 提取数据中当前帧odemotry输出的线速度和角速度
    u_t = [Robots{robot_num}.O(i, 2); Robots{robot_num}.O(i, 3)];

    rot = deltaT * u_t(2);
    halfRot = rot / 2;%不知道为什么这要除于2，好像不除也差不多的效果
    trans = u_t(1) * deltaT;
    
    % calculate the movement Jacobian
    G_t = [1 0 trans * -sin(theta + halfRot);
           0 1 trans * cos(theta + halfRot);
           0 0 1];
    % calculate motion covariance in control space
    M_t = [(alphas(1) * abs(u_t(1)) + alphas(2) * abs(u_t(2)))^2 0;
           0 (alphas(3) * abs(u_t(1)) + alphas(4) * abs(u_t(2)))^2];
    % calculate Jacobian to transform motion covariance to state space
    V_t = [cos(theta + halfRot) -0.5 * sin(theta + halfRot);
           sin(theta + halfRot) 0.5 * cos(theta + halfRot);
           0 1];
    
    % calculate pose update from odometry
    poseUpdate = [trans * cos(theta + halfRot);
                  trans * sin(theta + halfRot);
                  rot];
    
    poseMeanBar = poseMean + poseUpdate;
    poseCovBar = G_t * poseCov * G_t' + V_t * M_t * V_t';
    
    % get measurements
    % 这就是通过查找与当前时间点对应上的measurement，如果当前时间点
    % 没有观察到任何landmark，则返回值z就是一个空的3x1的vector
    % 如果观察到了landmark，则会把对应landmark的[x;y;ID]存放到z中
    % z的维度为3 x k 其中k就是当前观察到的landmark数量（没有观察到是一个特例）
    [z, measurementIndex] = getObservations(Robots, robot_num, t, measurementIndex, codeDict);
    
    % remove landmark ID from measurement because
    % we're trying to live without that
    % 因为我们这个代码是假设不知道相关性，也就是观察到的landmark是不知道其id的
    % 所以下面就把z的第三行数据给抹掉，然后利用极大似然估计去计算每一个landmark对应的id
    
   
%     z = z(1:2,:);
    
    % if features are observed
    % loop over all features and compute Kalman gain
    % 如果当前时刻没有观察到任何的landmark，update环节就会直接跳过
    % 如果当前是时刻观察到了landmark则进入update环节
    if z(1,1) > 0.1
        % 其中size(z,2)返回的是当前时刻观察到的landmark的数量
        for k = 1:size(z, 2) % loop over every observed landmark 
            
            % loop over all landmarks and compute MLE correspondence
            % 这里一次性初始化一个n层矩阵，每一层存放这到指定landmark对应的Z,S,H
            %
            predZ = zeros(n_landmarks, 1, 2); % predicted measurements
            predS = zeros(n_landmarks, 2, 2); % predicted measurement covariances
            predH = zeros(n_landmarks, 2, 3); % predicted measurement Jacobians
            max_prob = 0; 
            landmarkIndex = 0;
            % 开始使用极大似然估计算法找到和当前测量值z对应的landmark
            % 方法就是遍历所有的landmark，并且找到一个与当前测量值z最近的一个
            % 如何找到就是靠极大似然法,求概率值
            for j = 1:n_landmarks
                xDist = Landmark_Groundtruth(j, 2) - poseMeanBar(1);
                yDist = Landmark_Groundtruth(j, 3) - poseMeanBar(2);
                q = xDist^2 + yDist^2;
                
                % calculate predicted measurement
                predZ(j,:,:) = [sqrt(q);
                            conBear(atan2(yDist, xDist) - poseMeanBar(3))];
                        
                % calculate predicted measurement Jacobian
                predH(j,:,:) = [-xDist/sqrt(q) -yDist/sqrt(q) 0;
                                yDist/q        -xDist/q      -1];
                            
                % calculate predicted measurement covariance
                predS(j,:,:) = squeeze(predH(j,:,:)) * poseCovBar ...
                               * squeeze(predH(j,:,:))' + Q_t;
                
                % calculate probability of measurement correspondence     
                %  极大似然估计算法  
                probability = det(2 * pi * squeeze(predS(j,:,:)))^(-0.5) * ...
                        exp(-0.5 * (z(1:2,k) - squeeze(predZ(j,:,:)))' ...
                        * inv(squeeze(predS(j,:,:))) ...
                        * (z(1:2,k) - squeeze(predZ(j,:,:))));
                
                % update correspondence if the probability is
                % higher than the previous maximum
                % 保存最大的probalitity 和 对应的landmarkID,
                % 之后还要重新利用这个landmarkid 找到对应的 H S, 算一次update
                if probability > max_prob
                    max_prob = probability;
                    landmarkIndex = j;
                end
            end
            
%            检验是否极大似然的算法是正确的对比真实值id
            % TODO
            % 发现极大似然找到的id是和真实值有差异的,并且极大似然的结果更好. 为什么呢???因为真实值id不准确吗??

             true_id = z(3,k);
             temp = [true_id; landmarkIndex];
             IDS_comp = [IDS_comp  temp];
%             compute Kalman gain
%             利用上面极大似然找到的最佳的landmark ID 算kalman gain
            K = poseCovBar * squeeze(predH(landmarkIndex,:,:))' ...
                * inv(squeeze(predS(landmarkIndex,:,:)));

            % update pose mean and covariance estimates
            % 当前时刻可能观察到不同数量的landmark,每一个landmark都要用来
            % 更新当前时刻的均值和协方差,并且每一个当前landmark更新后的结果会
            % 传到下一个landmark进一步更新, 比如当前时刻t 看到了两个landmark,也就是k = 1:2
            % 分别为1 和2, 首先利用landmark1更新mean1和cov1,然后将更新之后的mean1和cov1
            % 传给landmark2进一步更新
            poseMeanBar = poseMeanBar + K * (z(1:2,k) - squeeze(predZ(landmarkIndex,:,:)));
            poseCovBar = (eye(3) - (K * squeeze(predH(landmarkIndex,:,:)))) * poseCovBar;
        end
    end
    
    % 将更新后的mean和cov传入下一帧
    % update pose mean and covariance
    poseMean = poseMeanBar;
    poseMean(3) = conBear(poseMean(3));
    poseCov = poseCovBar;
    
    % add pose mean to estimated position vector
    Robots{robot_num}.Est(i,:) = [t poseMean(1) poseMean(2) poseMean(3)];
    
    % calculate error between mean pose and groundtruth 
    % for testing only
    %{
    groundtruth = [Robots{robot_num}.G(i, 2); 
                   Robots{robot_num}.G(i, 3); 
                   Robots{robot_num}.G(i, 4)];
    error = groundtruth - poseMean;
    %}
   
end
% animateMRCLAMdataSet(Robots, Landmark_Groundtruth, timesteps, deltaT);