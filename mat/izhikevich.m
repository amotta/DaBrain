% Created by Eugene M. Izhikevich, February 25, 2003
% Available for download on the author's website:
% http://www.izhikevich.org/publications/spikes.htm
%
% Modified by Alessandro Motta and Hugo Zeberg

clear all
rng(0,'CombRecursive')

f=1;

% Excitatory neurons            Inhibitory neurons
% Number of different neurons
Ne = 800 * f;                   Ni = 200 * f;
% Random number to differentiate neurons
re = rand(Ne, 1);               ri = rand(Ni, 1);

% Neuronal parameters
a = [0.02 * ones(Ne,1);         0.02 + 0.08 * ri];
b = [0.2 * ones(Ne,1);          0.25 - 0.05 * ri];
% Membrane voltage reset
c = [-65 + 15 * re .^ 2;        -65 * ones(Ni,1)];
% Recovery variable reset
d = [8 - 6 * re .^ 2;           2 * ones(Ni,1)];

% Synapse matrix
S = [0.5 * rand(Ne + Ni, Ne),   -1.2 * rand(Ne + Ni, Ni)];

% Initial membrane voltage
v = -65 * ones(Ne + Ni, 1);
% Initial recovery value
u = b .* v;

%%
% Write data to CSV files
dynParam = [a'; b'; c'; d'];
csvwrite('dynParam.csv', dynParam);

dynState = [v'; u'];
csvwrite('dynState.csv', dynState);

%%
% Simulation
fired = [];
firings = [];
I = zeros(Ne + Ni, 1);
Isyn = zeros(Ne + Ni, 1);
Ithal = zeros(Ne + Ni, 1);

for t = 1 : 1000
    % find firing neurons
    fired = find(v >= 30);
    firings = [firings; t + 0 * fired, fired];
    
    % reset firing neurons
    v(fired) = c(fired);
    u(fired) = u(fired) + d(fired);
    
    % compute current
    Isyn = sum(S( : , fired), 2);
    Ithal = [5 * randn(Ne, 1); 2 * randn(Ni, 1)];
    I = Isyn + Ithal;
    
    % update state
    v = v + 0.5 * (0.04 * v .^ 2 + 5 * v + 140 - u + I);
    v = v + 0.5 * (0.04 * v .^ 2 + 5 * v + 140 - u + I);
    u = u + a .* (b .* v - u);
end

% plot firing neurons
plot(firings( : , 1), firings( : , 2), '.');