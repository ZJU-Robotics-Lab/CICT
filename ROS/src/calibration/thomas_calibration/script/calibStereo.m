
filename_pattern = '/Users/tangli/Desktop/201807191600/%s/%s_%02d.png';

num = 141;
left_filenames = {};
right_filenames = {};
for i = 0:(num-1)
  left_filenames{end+1} = sprintf(filename_pattern, 'left', 'left', i);
  right_filenames{end+1} = sprintf(filename_pattern, 'right_rotated', 'right', i);
end

%%
Itemp = imread(left_filenames{1});
h = size(Itemp, 1);
w = size(Itemp, 2);
imageSize = [w, h];

%%
squareSize = 0.1;
xSize = 6;
ySize = 7;
x_obj = repmat(((1:xSize) - 1)', 1, ySize);
y_obj = repmat(((1:ySize) - 1), xSize, 1);
z_obj = zeros(ySize, xSize);
boardPoints = squareSize * [x_obj(:), y_obj(:), z_obj(:)];
% boardPoints = num2cell(boardPoints, 2);
% boardPoints = reshape(boardPoints, 1, 42);

%%
[imagePoints, boardSize, imagesUsed] = ...
  detectCheckerboardPoints(left_filenames, right_filenames);
fprintf('detectCheckerboardPoints done\n');

%%
% imagePoints1 = num2cell(permute(imagePoints(:,:,:,1), [3, 1, 2]), 3);
% imagePoints2 = num2cell(permute(imagePoints(:,:,:,2), [3, 1, 2]), 3);
% objectPoints = repmat(boardPoints, [size(imagePoints, 3), 1]);
nImages = size(imagePoints, 3);
nPoints = size(imagePoints, 1);
imagePoints1 = cell(nImages, 1);
imagePoints2 = cell(nImages, 1);
objectPoints = cell(nImages, 1);
for i = 1:nImages
  imagePoints1{i} = cell(nPoints, 1);
  imagePoints2{i} = cell(nPoints, 1);
  objectPoints{i} = cell(nPoints, 1);
  for j = 1:nPoints
    imagePoints1{i}{j} = imagePoints(j, :, i, 1);
    imagePoints2{i}{j} = imagePoints(j, :, i, 2);
    objectPoints{i}{j} = boardPoints(j, :);
  end
end

% imagePoints1 = permute(imagePoints(:,:,:,1), [3, 1, 2]);
% imagePoints1 = permute(imagePoints(:,:,:,2), [3, 1, 2]);
% imagePoints1 = mat2cell(imagePoints1, ones(1, size(imagePoints1,1)), ones(1, size(imagePoints1, 2)), size(imagePoints1, 3));

%%
figure(1);
for i = 1:size(imagePoints1,1)
  
  clf;
  
  for j = 1:size(imagePoints1{i},1)
    
    color = [i / size(imagePoints1,1), j / size(imagePoints1{i},1), 0];
    
    p1 = imagePoints1{i}{j};
    subplot(1, 3, 1);
    plot(p1(1), p1(2), 'o', 'Color', color);  hold on;
    
    p2 = imagePoints2{i}{j};
    subplot(1, 3, 2);
    plot(p2(1), p2(2), 'o', 'Color' ,color);  hold on;
    
    p3 = objectPoints{i}{j};
    subplot(1, 3, 3);
    plot(p3(1), p3(2), 'o', 'Color' ,color);  hold on;
  end
  subplot(1, 3, 1);
  plot(imagePoints1{i}{1}(1), imagePoints1{i}{1}(2), 'go', 'MarkerSize', 10);
  
  subplot(1, 3, 2);
  plot(imagePoints2{i}{1}(1), imagePoints2{i}{1}(2), 'go', 'MarkerSize', 10);
  
  subplot(1, 3, 3); axis equal;
  plot(objectPoints{i}{1}(1), objectPoints{i}{1}(2), 'go', 'MarkerSize', 10);
  
  drawnow;
  waitforbuttonpress;
end


%%
[cameraMatrix1, distCoeffs1, reprojErr1, rvecs1, tvecs1, ~, ~, ~] = ...
  cv.calibrateCamera(objectPoints, imagePoints1, imageSize);
[cameraMatrix2, distCoeffs2, reprojErr2, rvecs2, tvecs2, ~, ~, ~] = ...
  cv.calibrateCamera(objectPoints, imagePoints2, imageSize);
fprintf('calib done\n');

% for i = 1:N

%%
S = cv.stereoCalibrate(...
  objectPoints, imagePoints1, imagePoints2, imageSize, ...
  'CameraMatrix1', cameraMatrix1, 'DistCoeffs1', distCoeffs1, ...
  'CameraMatrix2', cameraMatrix2, 'DistCoeffs2', distCoeffs2, ...
  'Criteria', struct('type','Count+EPS', 'maxCount',60, 'epsilon',1e-6))


%%
% SR = cv.stereoRectify(cameraMatrix1, distCoeffs1, ...
%   cameraMatrix2, distCoeffs2, imageSize, S.R, S.T)

%%
% R = [-1 0 0; 0 -1 0; 0 0 1] * S.R;%  * [-1 0 0; 0 -1 0; 0 0 1];
% T = [-1 0 0; 0 -1 0; 0 0 1] * S.T;
% R = S.R * [-1 0 0; 0 -1 0; 0 0 1];
% T = S.T;
R = S.R;
T = S.T;

SR = cv.stereoRectify(cameraMatrix1, distCoeffs1, ...
  cameraMatrix2, distCoeffs2, imageSize, R, T);

% SR.R2 = SR.R2 * [-1 0 0; 0 -1 0; 0 0 1];
% SR.R2 = [-1 0 0; 0 -1 0; 0 0 1] * SR.R2;

[map_l1, map_l2] = cv.initUndistortRectifyMap(cameraMatrix1, ...
  distCoeffs1, imageSize, 'R', SR.R1, 'NewCameraMatrix', SR.P1);
[map_r1, map_r2] = cv.initUndistortRectifyMap(cameraMatrix2, ...
  distCoeffs2, imageSize, 'R', SR.R2, 'NewCameraMatrix', SR.P2);

% cvParams = S;
%%
Il = imread(left_filenames{20});
Ir = imread(right_filenames{20});
Ilrect = cv.remap(Il, map_l1, map_l2, 'Interpolation', 'Cubic');
Irrect = cv.remap(Ir, map_r1, map_r2, 'Interpolation', 'Cubic');


param.disp_min    = 0;           % minimum disparity (positive integer)
param.disp_max    = 255;         % maximum disparity (positive integer)
param.subsampling = 0;
[D1,D2] = elasMex(rgb2gray(Ilrect), rgb2gray(Irrect), param);
 %figure(1); clf; imshow(stereoAnaglyph(Ilrect,Irrect));
% I = cat(2, Ilrect, imrotate(Irrect, 180));
I = cat(2, Ilrect, Irrect);

figure(2);
imshow(I);
for i=1:50:size(Ilrect,1)
  line([0, 2* size(Ilrect,2)], [i, i]);
end
figure(3); clf; imagesc(D1);

%%
D = disparity(rgb2gray(Ilrect), rgb2gray(Irrect), 'DisparityRange', [0, 256], 'UniquenessThreshold', 25);
D(D<0) = nan;
figure(4); imagesc(D);
%%
bm = cv.StereoSGBM('MinDisparity',25);% , 'MaxDisparity', 255);
bm.BlockSize = 15;
bm.SpeckleRange = 2;
bm.SpeckleWindowSize = 400;
bm.NumDisparities = 128;
bm.Mode = 0;
bm.UniquenessRatio = 5;
bm.PreFilterCap = 100;
bm.Disp12MaxDiff = 1;
bm.P1 = 8 * 15 * 15;
bm.P2 = 32 * 15 * 15;
D = bm.compute(rgb2gray(Ilrect), rgb2gray(Irrect));
D(D<0) = nan;
figure(5); imagesc(D/16);


%%
cvS = struct();
cvS.image_width = w;
cvS.image_height = h;

%%
cvS.camera_name = 'left';
cvS.camera_matrix = cameraMatrix1;
cvS.distortion_model = 'plumb_bob';
cvS.distortion_coefficients = distCoeffs1;
cvS.rectification_matrix = SR.R1;
cvS.projection_matrix = SR.P1;
cv.FileStorage(filename1, cvS);

%%
cvS.camera_name = 'right';
cvS.camera_matrix = cameraMatrix2;
cvS.distortion_coefficients = distCoeffs2;
cvS.rectification_matrix = SR.R2;
cvS.projection_matrix = SR.P2;
cv.FileStorage(filename2, cvS);

