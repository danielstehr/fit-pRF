function drawprf(x,y,sigma,n,linestyle)
% drawprf(x,y,sigma,n,linestyle)
% Draw pRF centered on (x,y) with radius equal to 2*(sigma/sqrt(n))
% <linestyle> can specify the color to draw line in
%       (default: 'r-')

if ~exist('linestyle','var') || isempty(linestyle)
  linestyle = 'r-';
end

hold on;

r = 2*(sigma/sqrt(n));
tt = 0:pi/50:2*pi;
xx = x + r*cos(tt);
yy  = y + r*sin(tt);

plot(xx,yy,linestyle);

hold off;

end

