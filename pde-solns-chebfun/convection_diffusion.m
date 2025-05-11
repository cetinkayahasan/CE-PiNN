clear all
clc
close all
%%%
% Modified for convection-diffusion equation
% u_t + c*u_x = D*u_xx

c = 3.0;      % Convection speed parameter
D = 0.1;      % Diffusion coefficient

nn = 511;
steps = 200;

dom = [0 2*pi]; x = chebfun('x',dom); tspan = linspace(0,1,steps+1);
S = spinop(dom, tspan);
% Convection-diffusion operator
S.lin = @(u) -c*diff(u,1) + D*diff(u,2);  % Convection (first derivative) and Diffusion (second derivative)
S.nonlin = @(u) 0*u;                       % No nonlinear term
S.init = exp(-(x-pi).^2/(2*(pi/4)^2));

u = spin(S, nn, 1e-4, 'plot', 'off');

usol = zeros(nn,steps+1);
for i = 1:steps+1
    usol(:,i) = u{i}.values;
end

x = linspace(0,2*pi,nn+1);
usol = [usol;usol(1,:)];
t = tspan;
pcolor(t,x,usol); shading interp, axis tight, colormap(jet);
title('Convection-Diffusion Equation Solution');
xlabel('Time');
ylabel('Space');

fname = "convection_diffusion_c" + num2str(c, '%.1f') + "_D" + num2str(c, '%.1f') + ".mat";
dfile = "./data/" + fname;
save(dfile, 't', 'x', 'usol');