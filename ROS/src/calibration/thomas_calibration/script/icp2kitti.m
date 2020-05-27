function icp2kitti(icpFile, outputDir)

fid = fopen(icpFile);
icp = fscanf(fid, '%s %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f', [(20+16) inf]);
fclose(fid);
timestamps = char(icp(1:16, :))';
poses = icp(21:end, :)';
fid = fopen(fullfile(outputDir, 'icp', 'timestamps.txt'), 'w');
fprintf(fid, '%c', [timestamps'; repmat(char(10), 1, size(timestamps,1))]);
fclose(fid);
fid = fopen(fullfile(outputDir, 'icp', 'icp.txt'), 'w');
fprintf(fid, '%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f\n', poses');
fclose(fid);