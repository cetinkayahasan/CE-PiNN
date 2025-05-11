clear all
clc
close all
%%%
% https://github.com/maziarraissi/DeepHPMs/blob/master/Matlab/gen_data_KS.m
% https://github.com/chebfun/examples/blob/master/pde/Kuramoto.m

nn = 511; % 511 100
steps = 100; % 200 100
l = 0.001;
rho = 10.0;

dom = [-1 1]; x = chebfun('x',dom); tspan = linspace(0,1,steps+1);
S = spinop(dom, tspan);
S.lin = @(u) + l*diff(u,2);
S.nonlin = @(u) -rho*u.*(u.^2 - 1); 
S.init = x^2*cos(pi*x);
%S.init = x^2*sin(2*pi*x);
u = spin(S, nn, 1e-4,'plot','off');

usol = zeros(nn,steps+1);
for i = 1:steps+1
    usol(:,i) = u{i}.values;
end

x = linspace(-1, 1,nn+1);
usol = [usol;usol(1,:)];
t = tspan;
pcolor(t,x,usol); shading interp, axis tight, colormap(jet);
usol = transpose(usol);
fname = "ac_l" + l + "_rho" + rho + ".mat";
%dfile = "./data/" + fname;
dfile = "./data-ac/" + fname;
save(dfile,'t','x','usol')