function writeMat(fid, name, mat)

fprintf(fid, '%s:', name);
fprintf(fid, ' %f', mat');
fprintf(fid, '\n');
