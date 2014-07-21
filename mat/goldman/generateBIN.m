% Generate file
clear all;

% Proportion of inhibitory neurons
pInh = 0.2;

% Upper and lower diagonals
ku = 500;
kl = 500;

% Number of neurons
N = 1024;
Ni = round(pInh * N);
Ne = N - Ni;

% Index of inhibitory neurons
posI = round(linspace(1, N, Ni));

% Random number to differentiate neurons
r = rand(N, 1);
ri = rand(Ni, 1);

% Leakage
gL = 1 / 4.3E9 * (0.95 + 0.1 * r);

% Na channel density
pNa = 20E-6 * (0.95 + 0.1 * r);
pNa(posI) = 20E-6 * (0.95 + 0.1 * ri);

% K channel density
pK = 3E-6 * (0.95 + 0.1 * r);
pK(posI) = 10E-6 * (0.95 + 0.1 * ri);

% The ECE 2014 poster compared the power spectra
% of the following two situations:
% pK = 0.9 * pK (blocking of Nv channels)
% pK = 1.1 * pK (enhancement of Nv channels)

% Neuron type
nType = 0 * ones(N, 1);
nType(posI) = 1 * ones(Ni, 1);

% Write data to file
dynParam = [gL, pNa, pK, nType];

fileID = fopen('dynParam.bin', 'w');
fwrite(fileID, size(dynParam), 'int');
fwrite(fileID, dynParam, 'float');
fclose(fileID);

% Membrane voltage
vZero = -70E-3 * ones(N, 1) + 30E-3 * rand(N, 1);

% Transmembrane current
% This value is never read
iZero = 0 * ones(N, 1);

% Na activation
mZero = zeros(N, 1);

% Na inactivation
hZero = ones(N, 1);

% K activation
nZero = zeros(N, 1);

% Write initial state
dynState = [vZero, iZero, mZero, hZero, nZero];

fileID = fopen('dynState.bin', 'w');
fwrite(fileID, size(dynState), 'int');
fwrite(fileID, dynState, 'float');
fclose(fileID);

% banded synapse matrix
synExc = zeros(ku + kl + 1, N);
synInh = zeros(ku + kl + 1, N);

% upper diagonals
for i = 1 : ku
	piv = (ku + 1) - i;
	pre = zeros(1, piv);
	suf = (i / ku) * rand(1, N - piv);
	synExc(i, : ) = [pre, suf];
end

% lower diagonals
for i = (ku + 2) : (ku + kl + 1)
	piv = i - (ku + 1);
	pre = (kl + 1 - piv) / kl * rand(1, N - piv);
	suf = zeros(1, piv);
	synExc(i, : ) = [pre, suf];
end

% inhibitory neurons
for bRow = 1 : (ku + 1 + kl)
    for bCol = posI
        if(bRow < max(1, ku + 2 - bCol))
            continue;
        elseif(bRow > min(ku + 1 + kl, ku + 1 + N - bCol))
            continue;
        else
            synExc(bRow, bCol) = 0;
            synInh(bRow, bCol) = 1;
        end
    end
end

% Scale matrix
synExc = 9E-9 * synExc;

% Write excitatory synapses
fileID = fopen('synExc.bin', 'w');
fwrite(fileID, size(synExc), 'int');
fwrite(fileID, synExc, 'float');
fclose(fileID);

% Inhibitory synapses
% synInh = 9E-11 * synInh;
synInh = 4.6E-9 * synInh;

fileID = fopen('synInh.bin', 'w');
fwrite(fileID, size(synInh), 'int');
fwrite(fileID, synInh, 'float');
fclose(fileID);

% Synapse parameter
synParam = [
    exp(-1/20), exp(-1/20);
    exp(-1/2), exp(-1/2)
];

fileID = fopen('synParam.bin', 'w');
fwrite(fileID, size(synParam), 'int');
fwrite(fileID, synParam, 'float');
fclose(fileID);

% Synapse state
synState = zeros(N, 6);

fileID = fopen('synState.bin', 'w');
fwrite(fileID, size(synState), 'int');
fwrite(fileID, synState, 'float');
fclose(fileID);

%% Run CUDA simulation
clear;

% run simulation
system('./dabrain');

%% Find firing neurons
load('firing.log');
[rows, cols] = find(firing(:, 2:end) > -0.035);

figure;
plot(rows, cols,'.');
title('Neuron firing');