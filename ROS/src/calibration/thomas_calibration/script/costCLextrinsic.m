function cost = costCLextrinsic(theta,boardPlane,boardPC)
cost = [];
R = eul2rotm(theta(4:6).');
t = theta(1:3);
for i = 1:size(boardPlane,2)
    pts = R*boardPC{i}+repmat(t,1,size(boardPC{i},2));
    cost = [cost boardPlane(1:3,i).'*pts+boardPlane(4,i)];
end
cost = cost*cost.';
end