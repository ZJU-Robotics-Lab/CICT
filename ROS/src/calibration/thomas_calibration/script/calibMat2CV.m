function calibMat2CV(stereoParams, filename1, filename2)

%%
cameraMatrix1 = stereoParams.CameraParameters1.IntrinsicMatrix';
distCoeffs1 = [stereoParams.CameraParameters1.RadialDistortion stereoParams.CameraParameters1.TangentialDistortion];

cameraMatrix2 = stereoParams.CameraParameters2.IntrinsicMatrix';
distCoeffs2 = [stereoParams.CameraParameters2.RadialDistortion stereoParams.CameraParameters2.TangentialDistortion];

imageSize = fliplr(stereoParams.CameraParameters1.ImageSize);
R = stereoParams.RotationOfCamera2;
T = stereoParams.TranslationOfCamera2';

SR = cv.stereoRectify(cameraMatrix1, distCoeffs1, ...
  cameraMatrix2, distCoeffs2, imageSize, R, T);

%%
S = struct();
S.image_width = stereoParams.CameraParameters1.ImageSize(2);
S.image_height = stereoParams.CameraParameters1.ImageSize(1);

%%
S.camera_name = 'left';
S.camera_matrix = cameraMatrix1;
S.distortion_model = 'plumb_bob';
S.distortion_coefficients = distCoeffs1;
S.rectification_matrix = SR.R1;
S.projection_matrix = SR.P1;
cv.FileStorage(filename1, S);

%%
S.camera_name = 'right';
S.camera_matrix = cameraMatrix2;
S.distortion_coefficients = distCoeffs2;
S.rectification_matrix = SR.R2;
S.projection_matrix = SR.P2;
cv.FileStorage(filename2, S);
