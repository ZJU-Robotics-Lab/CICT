function calib2kitti()

%%
input_dir = '/mnt/DataBlock2/thomas/YQ-south';
output_dir = '/mnt/DataBlock2/thomas/YQ-south_RAW';

%%
% CLEmain;
fid = fopen(fullfile(input_dir, 'calibration', 'calib_velo_to_cam.txt'), 'w');
fprintf(fid, 'calib_time: %s\n', 'unknown');
T = [Pose_CL;0 0 0 1]^(-1);
writeMat(fid, 'R', T(1:3,1:3));
writeMat(fid, 'T', T(1:3,4));
fclose(fid);

%%
calib = readCalibrationFromOpenCV(...
  fullfile(input_dir, 'calibration', 'left.yaml'), ...
  fullfile(input_dir, 'calibration', 'right.yaml'));

fid = fopen(fullfile(input_dir, 'calibration', 'calib_cam_to_cam.txt'), 'w');
fprintf(fid, 'calib_time: %s\n', 'unknown');
fprintf(fid, 'corner_dist: %f\n', 0.1);
writeMat(fid, 'S_00', calib.S_{1});
writeMat(fid, 'K_00', calib.K_{1});
writeMat(fid, 'D_00', calib.D_{1});
writeMat(fid, 'R_00', calib.R_{1});
writeMat(fid, 'T_00', calib.T_{1});
writeMat(fid, 'S_rect_00', calib.S_rect_{1});
writeMat(fid, 'R_rect_00', calib.R_rect_{1});
writeMat(fid, 'P_rect_00', calib.P_rect_{1});
writeMat(fid, 'S_01', calib.S_{2});
writeMat(fid, 'K_01', calib.K_{2});
writeMat(fid, 'D_01', calib.D_{2});
writeMat(fid, 'R_01', calib.R_{2});
writeMat(fid, 'T_01', calib.T_{2});
writeMat(fid, 'S_rect_01', calib.S_rect_{2});
writeMat(fid, 'R_rect_01', calib.R_rect_{2});
writeMat(fid, 'P_rect_01', calib.P_rect_{2});
fclose(fid);

end

