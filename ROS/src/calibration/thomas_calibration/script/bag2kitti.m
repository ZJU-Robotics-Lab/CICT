function bag2kitti(bag, output_dir, calib)

[map_l1, map_l2] = cv.initUndistortRectifyMap(...
  calib.K_{1}, calib.D_{1}, calib.P_rect_{1}, calib.S_rect_{1}, 'R', calib.R_rect_{1});
[map_r1, map_r2] = cv.initUndistortRectifyMap(...
  calib.K_{2}, calib.D_{2}, calib.P_rect_{2}, calib.S_rect_{2}, 'R', calib.R_rect_{2});

if exist(output_dir, 'dir')
  % fprintf('Ignoring %s\n', output_dir);
  % return;
end

mkdir(fullfile(output_dir, 'image_00', 'data'));
mkdir(fullfile(output_dir, 'image_01', 'data'));
mkdir(fullfile(output_dir, 'velodyne_points', 'data'));
mkdir(fullfile(output_dir, 'icp', 'data'));
mkdir(fullfile(output_dir, 'gps', 'data'));
mkdir(fullfile(output_dir, 'imu', 'data'));

if ischar(bag)
  fprintf('Reading ROS bag file...');
  bag = rosbag(bag);
  fprintf('done\n');
end

velo_msgs = bag.select('Topic', '/velodyne_points');

image00_msgs = bag.select('Topic', '/pointgrey/left/image_raw/compressed');
image01_msgs = bag.select('Topic', '/pointgrey/right/image_raw/compressed');


velo_ts_fid = fopen(fullfile(output_dir, 'velodyne_points', 'timestamps.txt'), 'w');
image00_ts_fid = fopen(fullfile(output_dir, 'image_00', 'timestamps.txt'), 'w');
image01_ts_fid = fopen(fullfile(output_dir, 'image_01', 'timestamps.txt'), 'w');

for i = 1:size(velo_msgs.MessageList,1)
  fprintf('\rHandling %d/%d...', i, size(velo_msgs.MessageList,1));
  
  velo_msg = readMessages(velo_msgs, i);
  velo_msg = velo_msg{1};
  
  [~, ind] = min(abs(velo_msgs.MessageList.Time(i) - image00_msgs.MessageList.Time));
  image00_msg = readMessages(image00_msgs, ind);
  image00_msg = image00_msg{1};
  % image00 = readImage(image00_msg{1});
  image00_bayer = cv.imdecode(image00_msg.Data, 'Flags', -1);
  image00 = cv.cvtColor(image00_bayer, 'BayerBG2RGB');
  
  [~, ind] = min(abs(velo_msgs.MessageList.Time(i) - image01_msgs.MessageList.Time));
  image01_msg = readMessages(image01_msgs, ind);
  image01_msg = image01_msg{1};
  image01_bayer = cv.imdecode(image01_msg.Data, 'Flags', -1);
  image01 = cv.cvtColor(image01_bayer, 'BayerBG2RGB');
  
  image00 = cv.remap(image00, map_l1, map_l2, 'Interpolation', 'Cubic');
  image01 = cv.remap(image01, map_r1, map_r2, 'Interpolation', 'Cubic');
  
  
  fid = fopen(fullfile(output_dir, 'velodyne_points', 'data', [num2str(i-1, '%010d') '.bin']), 'w');
  fwrite(fid, velo_msg.Data);
  fclose(fid);
  
  imwrite(image00, fullfile(output_dir, 'image_00', 'data', [num2str(i-1, '%010d') '.png']));
  imwrite(image01, fullfile(output_dir, 'image_01', 'data', [num2str(i-1, '%010d') '.png']));
  
  fprintf(velo_ts_fid, '%d.%d\n', velo_msg.Header.Stamp.Sec, velo_msg.Header.Stamp.Nsec);
  fprintf(image00_ts_fid, '%s\n', image00_msg.Header.Stamp.Sec, image00_msg.Header.Stamp.Nsec);
  fprintf(image01_ts_fid, '%s\n', image01_msg.Header.Stamp.Sec, image01_msg.Header.Stamp.Nsec);
  
end

fclose(velo_ts_fid);
fclose(image00_ts_fid);
fclose(image01_ts_fid);
