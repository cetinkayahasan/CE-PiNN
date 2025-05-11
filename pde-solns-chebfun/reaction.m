clear all
clc
close all
%%%
% Modified for reaction equation (Fisher-KPP type)
% u_t = r*u*(1 - u) 

r = 20.0;  % Reaction rate parameter

nn = 511;
steps = 200;

dom = [0 2*pi]; x = chebfun('x',dom); tspan = linspace(0,1,steps+1);
S = spinop(dom, tspan);
% Reaction term only (no diffusion/convection)
S.lin = @(u) 0*u;  % Zero linear operator
S.nonlin = @(u) r*u.*(1 - u);  % Reaction term (Fisher-KPP)
S.init = exp(-(x-pi).^2/(2*(pi/4)^2));
% S.init = 1 - sin(x);
u = spin(S, nn, 1e-4, 'plot', 'off');

usol = zeros(nn,steps+1);
for i = 1:steps+1
    usol(:,i) = u{i}.values;
end

x = linspace(0,2*pi,nn+1);
usol = [usol;usol(1,:)];
t = tspan;
pcolor(t,x,usol); shading interp, axis tight, colormap(jet);
title('Reaction Equation Solution (Fisher-KPP type)');
xlabel('Time');
ylabel('Space');

fname = "reaction_" + num2str(r,'%.1f') + ".mat";
dfile = "./data-reaction/" + fname;
save(dfile, 't', 'x', 'usol');