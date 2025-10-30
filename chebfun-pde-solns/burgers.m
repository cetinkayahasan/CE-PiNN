clear all
clc
close all

nn = 511; % 511 100
steps = 200; % 200 100

nu = 0.01/pi;

dom = [-1 1]; x = chebfun('x',dom); tspan = linspace(0,1,steps+1);
S = spinop(dom, tspan);
S.lin = @(u) + nu*diff(u,2);
S.nonlin = @(u) - 0.5*diff(u.^2); % spin cannot parse "u.*diff(u)"
S.init = -sin(pi*x);
%S.init = -sin(2*pi*x);
u = spin(S,nn,1e-5);

usol = zeros(nn,steps+1);
for i = 1:steps+1
    usol(:,i) = u{i}.values;
end

x = linspace(-1, 1,nn+1);
usol = [usol;usol(1,:)];
t = tspan;
pcolor(t,x,usol); shading interp, axis tight, colormap(jet);
usol = transpose(usol);
fname = "burgers_nu_" + nu + ".mat";
%dfile = "./data/" + fname;
dfile = "./data2/" + fname;
save(dfile,'t','x','usol')
