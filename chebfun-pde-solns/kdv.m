clear all
clc
close all

lambda2 = 0.0025; % not changing
lambda1 = 1.0;
nn = 511;
steps = 200;

dom = [-1 1]; x = chebfun('x',dom); tspan = linspace(0,1,steps+1);
S = spinop(dom, tspan);
S.lin = @(u) - lambda2*diff(u,3);
S.nonlin = @(u) - lambda1*1/2*diff(u.^2); 
S.init = cos(pi*x);
u = spin(S, nn, 1e-4,'plot','off');

usol = zeros(nn,steps+1);
for i = 1:steps+1
    usol(:,i) = u{i}.values;
end

x = linspace(-1, 1,nn+1);
usol = [usol;usol(1,:)];
t = tspan;
pcolor(t,x,usol); shading interp, axis tight, colormap(jet);
%save('kdv.mat', 't', 'x', 'usol')

usol = transpose(usol);
fname = "kdv_lambda1_" + num2str(lambda1,'%.1f')+ "_lambda2_" + lambda2 + ".mat";
dfile = "./data/" + fname;
save(dfile, 't', 'x', 'usol');
