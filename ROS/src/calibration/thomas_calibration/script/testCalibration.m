
%%
addpath('~/Projects/mexopencv/');

%%
% S = cv.stereoCalibrate(stereoParams.WorldPoints, imagePoints1, imagePoints2, imageSize);

%%
cameraMatrix1 = stereoParams.CameraParameters1.IntrinsicMatrix';
distCoeffs1 = [...
  stereoParams.CameraParameters1.RadialDistortion ...
  stereoParams.CameraParameters1.TangentialDistortion];

cameraMatrix2 = stereoParams.CameraParameters2.IntrinsicMatrix';
distCoeffs2 = [...
  stereoParams.CameraParameters2.RadialDistortion ...
  stereoParams.CameraParameters2.TangentialDistortion];

imageSize = fliplr(stereoParams.CameraParameters1.ImageSize);
R = stereoParams.RotationOfCamera2' * [-1 0 0; 0 -1 0; 0 0 1];
T = -stereoParams.TranslationOfCamera2'/1000;

S = cv.stereoRectify(cameraMatrix1, distCoeffs1, ...
  cameraMatrix2, distCoeffs2, imageSize, R, T);

S.R2 = [-1 0 0; 0 -1 0; 0 0 1] * S.R2;

[map_l1, map_l2] = cv.initUndistortRectifyMap(cameraMatrix1, ...
  distCoeffs1, imageSize, 'R', S.R1, 'NewCameraMatrix', S.P1);
[map_r1, map_r2] = cv.initUndistortRectifyMap(cameraMatrix2, ...
  distCoeffs2, imageSize, 'R', S.R2, 'NewCameraMatrix', S.P2);

% cvParams = S;
%%
Il = imread('/Users/tangli/Desktop/201807191600/left/left_50.png');
Ir = imread('/Users/tangli/Desktop/201807191600/right/right_50.png');
Ilrect = cv.remap(Il, map_l1, map_l2, 'Interpolation', 'Cubic');
Irrect = cv.remap(Ir, map_r1, map_r2, 'Interpolation', 'Cubic');
 %figure(1); clf; imshow(stereoAnaglyph(Ilrect,Irrect));
% I = cat(2, Ilrect, imrotate(Irrect, 180));
I = cat(2, Ilrect, Irrect);

figure(2);
imshow(I);
for i=1:50:size(Ilrect,1)
  line([0, 2* size(Ilrect,2)], [i, i]);
end

