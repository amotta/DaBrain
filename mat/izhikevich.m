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
S = [0.5 * rand(Ne + Ni, Ne),   -rand(Ne + Ni, Ni)];

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
% TODO
% Clean up code for simulation