% Generate file
clear all;

% Open file
fileID = fopen('config.bin', 'w');

% Proportion of inhibitory neurons
pInh = 0.2;

% Upper and lower diagonals
ku = 500;
kl = 500;

% Number of neurons
N = 10016;
Ni = round(pInh * N);
Ne = N - Ni;

% Write neuron count
fwrite(fileID, N, 'int');

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
fwrite(fileID, gL, 'float');
fwrite(fileID, pNa, 'float');
fwrite(fileID, pK, 'float');
fwrite(fileID, nType, 'float');

% Membrane voltage
vZero = -70E-3 * ones(N, 1);

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
fwrite(fileID, vZero, 'float');
fwrite(fileID, iZero, 'float');
fwrite(fileID, mZero, 'float');
fwrite(fileID, hZero, 'float');
fwrite(fileID, nZero, 'float');

% banded synapse matrix
bS = zeros(ku + kl + 1, N);

% upper diagonals
for i = 1 : ku
	piv = (ku + 1) - i;
	pre = zeros(1, piv);
	suf = (i / ku) * rand(1, N - piv);
	bS(i, : ) = [pre, suf];
end

% lower diagonals
for i = (ku + 2) : (ku + kl + 1)
	piv = i - (ku + 1);
	pre = (kl + 1 - piv) / kl * rand(1, N - piv);
	suf = zeros(1, piv);
	bS(i, : ) = [pre, suf];
end

% inhibitory neurons
for bRow = 1 : (ku + 1 + kl)
    for bCol = posI
        if(bRow < max(1, ku + 2 - bCol))
            continue;
        elseif(bRow > min(ku + 1 + kl, ku + 1 + N - bCol))
            continue;
        else
            bS(bRow, bCol) = -2 * bS(bRow, bCol);
        end
    end
end

% Scale matrix
bS = 2E-12 * bS;

% Write synapse matrix
fwrite(fileID, ku, 'int');
fwrite(fileID, kl, 'int');
fwrite(fileID, bS, 'float');
fwrite(fileID, bS, 'float');

%% Run CUDA simulation
system('./dabrain');
load('firing.log');

figure;
plot(firing(:, 1), firing(:, 2), '.');
title('Neuron firing CUDA');