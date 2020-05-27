function calib = readCalibrationFromOpenCV(left, right)

%%
LY = ReadYaml(left);
RY = ReadYaml(right);

%%
calib.S_{1} = [LY.image_width LY.image_height];
calib.K_{1} = reshape(cell2mat(LY.camera_matrix.data), 3, 3)';
calib.D_{1} = cell2mat(LY.distortion_coefficients.data);
calib.R_{1} = reshape(cell2mat(LY.rotation_matrix.data), 3, 3)';
calib.T_{1} = cell2mat(LY.translation_matrix.data)';

calib.S_{2} = [RY.image_width RY.image_height];
calib.K_{2} = reshape(cell2mat(RY.camera_matrix.data), 3, 3)';
calib.D_{2} = cell2mat(RY.distortion_coefficients.data);
calib.R_{2} = reshape(cell2mat(RY.rotation_matrix.data), 3, 3)';
calib.T_{2} = cell2mat(RY.translation_matrix.data)';

calib.S_rect_{1} = [LY.image_width LY.image_height];
calib.R_rect_{1} = reshape(cell2mat(LY.rectification_matrix.data), 3, 3)';
calib.P_rect_{1} = reshape(cell2mat(LY.projection_matrix.data), 4, 3)';
calib.S_rect_{2} = [RY.image_width RY.image_height];
calib.R_rect_{2} = reshape(cell2mat(RY.rectification_matrix.data), 3, 3)';
calib.P_rect_{2} = reshape(cell2mat(RY.projection_matrix.data), 4, 3)';

%% Determine rectified size
%{
I0 = ones(LY.image_height, LY.image_width, 'uint8') * 255;
need_update = true;
while need_update
    need_update = false;
    S = cv.stereoRectify(calib.K_{1}, calib.D_{1}, calib.K_{2}, calib.D_{2}, ...
        calib.S_rect_{1}, calib.R_{2}, calib.T_{2});
    calib.R_rect_{1} = S.R1;
    calib.P_rect_{1} = S.P1;
    calib.R_rect_{2} = S.R2;
    calib.P_rect_{2} = S.P2;
    [map_l1, map_l2] = cv.initUndistortRectifyMap(calib.K_{1}, calib.D_{1}, calib.P_rect_{1}, calib.S_rect_{1}, 'R', calib.R_rect_{1});
    I1 = cv.remap(I0, map_l1, map_l2, 'Interpolation', 'Cubic');
    [map_r1, map_r2] = cv.initUndistortRectifyMap(calib.K_{2}, calib.D_{2}, calib.P_rect_{2}, calib.S_rect_{2}, 'R', calib.R_rect_{2});
    I2 = cv.remap(I0, map_r1, map_r2, 'Interpolation', 'Cubic');
    
    figure(1); subplot(1,2,1); imagesc(I1); subplot(1,2,2); imagesc(I2);
    waitforbuttonpress;
    if ~isempty(find(~(I1&I2),1))
        calib.S_rect_{1} = [min(S.roi1(3), S.roi2(3)) min(S.roi1(4), S.roi2(4))]; % calib.S_rect_{1} - 1;
        calib.S_rect_{2} = calib.S_rect_{1};
        need_update = true;
    end
end
%}