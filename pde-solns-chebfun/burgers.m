clear all
clc
close all
%% I think this is the correct ref solns!!!!
%%% Parameters
nn = 511;       % Spatial resolution 400 3200  511
steps = 200;    % Time steps 200
nu = 0.005/pi;      % Viscosity parameter (similar to your l parameter in Allen-Cahn)

%%% Domain and time setup
dom = [-1 1]; 
x = chebfun('x',dom); 
tspan = linspace(0,1,steps+1);

%%% Set up the PDE using spinop
S = spinop(dom, tspan);
S.lin = @(u) nu*diff(u,2);            % Linear term (diffusion)
S.nonlin = @(u) -0.5*diff(u.^2,1);    % Nonlinear term (convection)
%S.init = -sin(pi*x); % Initial condition (common for Burgers')  
S.init = -sin(2*pi*x);

%%% Solve the PDE
u = spin(S, nn, 1e-4, 'plot', 'off');


%%% Prepare the solution for saving
usol = zeros(nn, steps+1);
for i = 1:steps+1
    usol(:,i) = u{i}.values;
end




x = linspace(-1, 1,nn+1);
usol = [usol;usol(1,:)];
t = tspan;

pcolor(t,x,usol); shading interp, axis tight, colormap(jet);
%usol = transpose(usol);
fname = "burgers_nu_" + nu*pi + ".mat";
dfile = "./data-burger/" + fname;
save(dfile, 't', 'x', 'usol');

