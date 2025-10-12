function dx = pendcart(x,m,M,L,g,d,u)
  Sx = sin(x(3));
  Cx = cos(x(3));
  D = m*L*L*(M+m*(1-Cx^2));
  dx(1,1) = x(2);
  dx(2,1) = (1/D)*(-m^2*L^2*g*Cx*Sx + m*L^2*(m*L*x(4)^2*Sx - d *x(2))) + m*L*L*(1/D)*u;
  dx(3,1) = x(4);
  dx(4,1) = (1/D)*((m+M)*m*g*L*Sx   - m*L*Cx*(m*L*x(4)^2*Sx - d* x(2)))- m*L*Cx*(1/D)*u;

clear all, close all, clc
m = 1;
M = 5;
L = 2;
g = -10;
d = 1;
b = 1; % Pendulum up (b=1)
A = [0          1               0  0;
     0       -d/M         b*m*g/M  0;
     0          0               0  1;
     0 -b*d/(M*L) -b*(m+M)*g/(M*L) 0];

B = [0; 1/M; 0; b/(M*L)];


%% Design LQR controller
Q = eye(4); % 4x4 identify matrix
R = .0001;
K = lqr(A,B,Q,R);

%% Simulate closed-loop system
tspan = 0:.001:10;
x0 = [-1; 0; pi+.1; 0]; % initial condition
wr = [1; 0; pi; 0]; % reference position
u=@(x)-K*(x - wr); % control law
[t,x] = ode45(@(t,x)pendcart(x,m,M,L,g,d,u(x)),tspan,x0);
