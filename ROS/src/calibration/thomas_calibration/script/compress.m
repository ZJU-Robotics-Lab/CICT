
source_dir = '/mnt/DataBlock2/thomas/YQ_RAW';
dest_dir = '/mnt/DataBlock2/thomas/YQ.compressed';

old_dir = pwd;
cd(source_dir);

date_dirs = lsdir('.');

for i=1:length(date_dirs)
  dirs = lsdir(fullfile(source_dir, date_dirs{i}));
  for j=1:length(dirs)
    fprintf('Compressing %s...\n', dirs{j});
    zip([dest_dir '/' dirs{j} '.zip'], fullfile(date_dirs{i}, dirs{j}))
  end
end

cd(old_dir);
