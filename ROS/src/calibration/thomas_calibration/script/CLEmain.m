close all;clear;clc;
%% initialization

dataDir = '/Users/tangli/Desktop/calib/space180601';
%  dataDir = 'E:\calib from kitti\2011_09_26_drive_0119_extract\2011_09_26\2011_09_26_drive_0119_extract\image_00\data\';
 
% dataDir = 'calib';
%Nimages = length(dir([dataDir '\*.png']));
Nimages = 53;
gridsize = 0.1;
baseline = 501.3124/1000;
marginthres = 0.1;   %0.1
depththres = 0.3;    %0.3
undistorted = true;
options = optimoptions('fminunc','PlotFcn',@optimplotfval);


%
% IntrinsicMatrix = [874.1081       0  643.1469;...
%                          0         876.9099  525.8110;...
%                          0        0         1];
IntrinsicMatrix = [836.7203        0   616.3885;...
                   0         836.7203  515.9765;...
                   0        0         1];
% IntrinsicMatrix = [854.646632        0  646.22031001;...
%                          0  854.681484  510.62377369;...
%                          0        0         1];
%{
IntrinsicMatrix = [411.3075        0  330.096;...
                         0  412.8864  167.9043;...
                         0        0         1];
%}
                     
%{
IntrinsicMatrix = [1402.06        0  1141.02;...
                         0  1402.06  608.297 ;...
                         0        0         1];
%}
%{
IntrinsicMatrix = [1376.03        0   1098.17;...
                         0  1375.41   614.903;...
                         0        0         1];
%}
if ~undistorted
    cameraParams = cameraParameters('IntrinsicMatrix',IntrinsicMatrix.','RadialDistortion',[-0.0929,0.0924],'WorldUnits','m');
else
    cameraParams = cameraParameters('IntrinsicMatrix',IntrinsicMatrix.','WorldUnits','m');
end
Pose_CL = [0  -1   0  baseline./2;...
           0   0  -1  -0.37;...
           1   0   0  0];

%% chessboard corners
fn_image = cell(0);
fn_cloud = cell(0);
for i = 0:Nimages-1
     fn_image{end+1} = [dataDir '/left_rect/left_rect_' num2str(i, '%02d') '.png'];
    fn_cloud{end+1} = [dataDir '/cloud_' num2str(i, '%02d') '.pcd'];
   
%     fn_cloud{end+1} = [dataDir '/cloud.' num2str(i) '.ply'];
    if ~undistorted
        I = imread([dataDir '/left/left_rect_' num2str(i) '.png']);
        I1 = undistortImage(I, cameraParams);
        imwrite(I1,[dataDir '/left/left_rect_' num2str(i) '.png'],'png')
    end
end

%%
[imagePoints,boardSize,pairsUsed] = detectCheckerboardPoints(fn_image);

if size(imagePoints,3)~=Nimages
    warning('Some image can not be detected with chessboard!');
   
%     mkdir([dataDir '/tmp']);
%     for i=1:Nimages
%         if ~pairsUsed(i)
%             movefile(fn_image{i}, [dataDir '/tmp']);
%             movefile(fn_cloud{i}, [dataDir '/tmp']);
%         end
%     end
    Nimages = size(imagePoints,3);
    fn_image = fn_image(pairsUsed);
    fn_cloud = fn_cloud(pairsUsed);
end

%% chessboard in world
worldPoints = [];
for i = 1:boardSize(2)-1
    for j =1:boardSize(1)-1
        worldPoints = [worldPoints;(i-1)*gridsize (j-1)*gridsize];
    end
end
worldNormal = [0 0 1].';

%% extrinsic camera
boardPose = zeros(3,4,Nimages);
boardPlane = zeros(4,Nimages);
for i = 1:Nimages
    [rotationMatrix,translationVector] = extrinsics(imagePoints(:,:,i),worldPoints,cameraParams);
    theta = [translationVector.';rotm2eul(rotationMatrix.').'];
    theta = fminunc(@(theta)costReproj(theta,imagePoints(:,:,i),[worldPoints zeros(size(worldPoints,1),1)],IntrinsicMatrix),theta);
    boardPose(1:3,1:3,i) = eul2rotm(theta(4:6).');
    boardPose(1:3,4,i) = theta(1:3);
    boardPlane(1:3,i) = boardPose(1:3,1:3,i)*worldNormal;
    boardPlane(4,i) = -boardPlane(1:3,i).'*boardPose(1:3,4,i);
    %{
    I = imread(fn_image{i});
    imshow(I);hold on;
    testimagePoints = boardPose(1:3,1:3,i)*[worldPoints.';zeros(1,size(worldPoints,1))]+repmat(boardPose(:,4,i),1,size(worldPoints,1));
    testimagePoints = IntrinsicMatrix*testimagePoints;
    plot(testimagePoints(1,:)./testimagePoints(3,:),testimagePoints(2,:)./testimagePoints(3,:),'ro');
    plot(imagePoints(:,1,i),imagePoints(:,2,i),'g*');hold off;
    waitforbuttonpress
    %}
end

%% extract plane
boardPC = cell(0);
boardPts_camera = [];
boardPts_laser = [];
for i = 1:Nimages
    I = imread(fn_image{i});
    lu = imagePoints(1,:,i);
    ru = imagePoints((boardSize(1)-1)*(boardSize(2)-2)+1,:,i);
    ld = imagePoints(boardSize(1)-1,:,i);
    rd = imagePoints(prod(boardSize-1),:,i);
    corner = [lu;ru;ld;rd];
    ubound = min(corner(:,2))-marginthres*size(I,1);
    lbound = min(corner(:,1))-marginthres*size(I,1);
    bbound = max(corner(:,2))+marginthres*size(I,2);
    rbound = max(corner(:,1))+marginthres*size(I,2);
    worldPointsc = boardPose(1:3,1:3,i)*[worldPoints.';zeros(1,size(worldPoints,1))]+repmat(boardPose(:,4,i),1,size(worldPoints,1));
    boardPts_camera = [boardPts_camera worldPointsc];
    zrange = [min(worldPointsc(3,:)) max(worldPointsc(3,:))];
    nearbound = zrange(1)-abs(diff(zrange))*depththres;
    farbound = zrange(2)+abs(diff(zrange))*depththres;
    % pts = importdata([dataDir '/cloud.' num2str(i-1) '.ply'],' ',30);
    pts = importdata(fn_cloud{i},' ',30);
    pts = pts.data(1:end-7,1:3).';
    ptsc = Pose_CL(1:3,1:3)*pts+repmat(Pose_CL(:,4),1,size(pts,2));
    ptscimgpts = IntrinsicMatrix*ptsc;
    ptscimgpts(1,:) = ptscimgpts(1,:)./ptscimgpts(3,:);
    ptscimgpts(2,:) = ptscimgpts(2,:)./ptscimgpts(3,:);
    valid = ptscimgpts(3,:)>nearbound & ptscimgpts(3,:)<farbound & ptscimgpts(1,:)>lbound & ...
        ptscimgpts(1,:)<rbound & ptscimgpts(2,:)>ubound & ptscimgpts(2,:)<bbound;
    ptsboardc = ptsc(:,valid);
    [~,inlier,~] = pcfitplane(pointCloud(ptsboardc.'),0.007);
    boardPC{end+1} = ptsboardc(:,inlier);
    boardPts_laser = [boardPts_laser ptsboardc(:,inlier)];
    
    %{
    cm = colormap(jet);
    %imshow(I);hold on;
    testvalid = ptscimgpts(3,:)>0 & ptscimgpts(3,:)<8 & ptscimgpts(1,:)>0 & ...
        ptscimgpts(1,:)<size(I,2) & ptscimgpts(2,:)>0 & ptscimgpts(2,:)<size(I,1);
    scatter(ptscimgpts(1,testvalid),ptscimgpts(2,testvalid),10,cm(round(ptscimgpts(3,testvalid)./8*size(cm,1)),:),'fill');
    plot3(ptsc(1,:),ptsc(2,:),ptsc(3,:),'r.','MarkerSize',0.5)
    axis equal;hold on;
    plot3(ptsboardc(1,inlier),ptsboardc(2,inlier),ptsboardc(3,inlier),'go');
    plot3(worldPointsc(1,:),worldPointsc(2,:),worldPointsc(3,:),'k.')
    hold off;
    
    while waitforbuttonpress ~= 1
    end
    
    %}
end

%% compute extrinsic
theta = zeros(6,1);
theta = fminunc(@(theta)costCLextrinsic(theta,boardPlane,boardPC),theta,options);
incrR = eul2rotm(theta(4:6).');
incrt = theta(1:3);
Pose_CL(1:3,1:3) = incrR*Pose_CL(1:3,1:3);
Pose_CL(1:3,4) = incrR*Pose_CL(1:3,4)+incrt;

%{
for i = 1:Nimages
    pts = importdata(fn_cloud{i},' ',30);
    pts = pts.data.';
    ptsc = Pose_CL(1:3,1:3)*pts+repmat(Pose_CL(:,4),1,size(pts,2));
    ptscimgpts = IntrinsicMatrix*ptsc;
    ptscimgpts(1,:) = ptscimgpts(1,:)./ptscimgpts(3,:);
    ptscimgpts(2,:) = ptscimgpts(2,:)./ptscimgpts(3,:);
    valid = ptscimgpts(3,:)>0 & ptscimgpts(3,:)<8 & ptscimgpts(1,:)>0 & ...
        ptscimgpts(1,:)<size(I,2) & ptscimgpts(2,:)>0 & ptscimgpts(2,:)<size(I,1);
    I = imread(fn_image{i});
    imshow(I);hold on;
    cm = colormap(jet);
    scatter(ptscimgpts(1,valid),ptscimgpts(2,valid),10,cm(round(ptscimgpts(3,valid)./8*size(cm,1)),:),'fill');
    waitforbuttonpress;
end
%}

%% 3d refinement
boardPts_laser_c = incrR*boardPts_laser + repmat(incrt,1,size(boardPts_laser,2));
tform = pcregrigid(pointCloud(boardPts_laser_c.'),pointCloud(boardPts_camera.'),'Metric','pointToPlane','InlierRatio',0.9,'Verbose',true);
T = tform.T.';
incrt = T(1:3,1:3)*incrt+T(1:3,4);
incrR = T(1:3,1:3)*incrR;
Pose_CL = T*[Pose_CL;0 0 0 1];
Pose_CL = Pose_CL(1:3,1:4);

%% check results
figure;
for i = 1:Nimages
    pts = importdata(fn_cloud{i},' ',30);
    pts = pts.data(1:end-7,1:3).';
    %pts = pts.data.';
    ptsc = Pose_CL(1:3,1:3)*pts+repmat(Pose_CL(:,4),1,size(pts,2));
    ptscimgpts = IntrinsicMatrix*ptsc;
    ptscimgpts(1,:) = ptscimgpts(1,:)./ptscimgpts(3,:);
    ptscimgpts(2,:) = ptscimgpts(2,:)./ptscimgpts(3,:);
    valid = ptscimgpts(3,:)>0 & ptscimgpts(3,:)<8 & ptscimgpts(1,:)>0 & ...
        ptscimgpts(1,:)<size(I,2) & ptscimgpts(2,:)>0 & ptscimgpts(2,:)<size(I,1);
    I = imread(fn_image{i});
    imshow(I);hold on;
    cm = colormap(jet);
    scatter(ptscimgpts(1,valid),ptscimgpts(2,valid),10,cm(round(ptscimgpts(3,valid)./8*size(cm,1)),:),'fill');hold off;
    waitforbuttonpress;
end

pose{1} = eye(4);
pose{2} = [Pose_CL;0 0 0 1];
drawPoses(pose)
