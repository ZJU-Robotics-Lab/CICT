
%%
addpath('~/Projects/YAMLMatlab/');
addpath(genpath('~/Projects/mexopencv/'));
addpath('~/Projects/robot_toolbox/');

%%
input_dir = '/mnt/DataBlock2/thomas/YQ-south';
output_dir = '/mnt/DataBlock2/thomas/YQ-south_RAW';

calib = readCalibrationFromOpenCV(...
  fullfile(input_dir, 'calibration', 'left.yaml'), ...
  fullfile(input_dir, 'calibration', 'right.yaml'));


%%
dirs = lsdir(input_dir);
for i=1:length(dirs)
  
  %
  date_dir = [dirs{i}(1:4) '_' dirs{i}(6:7) '_' dirs{i}(9:10)];
  if exist(fullfile(output_dir, date_dir), 'dir')
    fprintf('skipped: %s.\n', fullfile(output_dir, date_dir));
    continue;
  end
  mkdir(fullfile(output_dir, date_dir));
  
  % bags
  bagFileNames = dir(fullfile(input_dir, dirs{i}, '*.bag'));
  if ~exist('bags', 'var')
    bags = cell(length(bagFileNames), 1);
  end
  for j=1:length(bagFileNames)
    fprintf('Converting `%s` to `%s`...\n', ...
      fullfile(input_dir, dirs{i}, bagFileNames(j).name), ...
      fullfile(output_dir, date_dir, [date_dir '_drive_' num2str(j, '%04d') '_sync']));
    if isempty(bags{j})
      bags{j} = rosbag(fullfile(input_dir, dirs{i}, bagFileNames(j).name));
    end
    bag2kitti(bags{j}, ...
      fullfile(output_dir, date_dir, [date_dir '_drive_' num2str(j, '%04d') '_sync']), calib);
  end
  
  % icp results
  icpFileNames = dir(fullfile(input_dir, dirs{i}, '*.txt'));
  for j=1:length(icpFileNames)
    fprintf('Converting `%s` to `%s`...\n', ...
      fullfile(input_dir, dirs{i}, icpFileNames(j).name), ...
      fullfile(output_dir, date_dir, [date_dir '_drive_' num2str(j, '%04d') '_sync']));
    icp2kitti(...
      fullfile(input_dir, dirs{i}, icpFileNames(j).name), ...
      fullfile(output_dir, date_dir, [date_dir '_drive_' num2str(j, '%04d') '_sync']));
  end
end

