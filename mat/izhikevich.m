% Created by Eugene M. Izhikevich, February 25, 2003
% Available for download on the author's website:
% http://www.izhikevich.org/publications/spikes.htm
%
% Modified by Alessandro Motta and Hugo Zeberg

%% Import CSV files
clear all;
load('dynParam.csv');
a = dynParam(1, : )';
b = dynParam(2, : )';
c = dynParam(3, : )';
d = dynParam(4, : )';

load('dynState.csv');
v = dynState(1, : )';
u = dynState(2, : )';

load('syn.csv');
synRows = size(syn, 1);
N = size(syn, 2);
ku = (synRows - 1) / 2;
kl = (synRows - 1) / 2;

S = zeros(N);

% copy upper diagonals
for k = 1 : ku
    S = S + diag(syn(ku + 1 - k, (k + 1) : end), k);
end

for k = 1:kl
    S = S + diag(syn(ku + 1 + k, 1 : (end - k)), -k);
end

clear dynState;
clear dynParam;
clear syn;

%% Simulation
fired = [];
firings = [];
I = zeros(N, 1);
Isyn = zeros(N, 1);
Ithal = 5 * ones(N, 1);

for t = 1 : 1000
    % find firing neurons
    fired = find(v >= 30);
    firings = [firings; t + 0 * fired, fired];
    
    % Isyn = S * (v >= 30);
    
    % reset firing neurons
    v(fired) = c(fired);
    u(fired) = u(fired) + d(fired);
    
    % compute current
    Isyn = sum(S( : , fired), 2);
    I = Isyn + Ithal;
    
    % update state
    v = v + 0.5 * (0.04 * v .^ 2 + 5 * v + 140 - u + I);
    v = v + 0.5 * (0.04 * v .^ 2 + 5 * v + 140 - u + I);
    u = u + a .* (b .* v - u);
end

%% Plot firing
figure;
plot(firings( : , 1), firings( : , 2), '.');
title('Neuron firing MatLab');