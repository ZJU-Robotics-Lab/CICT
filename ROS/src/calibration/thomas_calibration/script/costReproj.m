function cost = costReproj(theta,imagePoints,worldPoints,IntrinsicMatrix)
R = eul2rotm(theta(4:6).');
t = theta(1:3);
pts = R*worldPoints.'+repmat(t,1,size(worldPoints,1));
testimagePoints = IntrinsicMatrix*pts;
testimagePoints = [testimagePoints(1,:)./testimagePoints(3,:);testimagePoints(2,:)./testimagePoints(3,:)];
residue = imagePoints- testimagePoints.';
cost = sum(residue(:).^2);
end