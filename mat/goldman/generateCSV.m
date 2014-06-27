% Generating all CSV files
clear all;

% Thousands of neurons
f = 0.01;

% Proportion of inhibitory neurons
pInh = 0.2;

% Upper and lower diagonals
ku = 1;
kl = 1;

N = round(f * 1000);
Ni = round(pInh * N);
Ne = N - Ni;

% Index of inhibitory neurons
posI = round(linspace(1, N, Ni));

% Random number to differentiate neurons
r = rand(N, 1);
ri = rand(Ni, 1);

% Leakage
gL = 1 / 4.3E9 * (1 + 0 * r);

% Na channel density
pNa = 20E-6 * (1 + 0 * r);
pNa(posI) = 20E-6 * (1 + 0 * ri);

% K channel density
pK = 3E-6 * (1 + 0 * r);
pK(posI) = 10E-6 * (1 + 0 * ri);

% Neuron type
nType = 0 * (1 + 0 * r);
nType(posI) = 1 * (1 + 0 * ri);

% Write data to CSV files
dynParam = [gL, pNa, pK, nType];
csvwrite('dynParam.csv', dynParam);

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

% Save initial states
dynState = [vZero, iZero, mZero, hZero, nZero];
csvwrite('dynState.csv', dynState);

% banded synapse matrix
bS = zeros(ku + kl + 1, N);

% upper diagonals
for i = 1 : ku
	piv = (ku + 1) - i;
	pre = zeros(1, piv);
	suf = (i / ku ) * rand(1, N - piv);
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
for i = 1 : (ku + 1 + kl)
    for j = posI
        bRow = ku + 1 + i - j;
        bCol = j;
        
        if(i < max(1, j - ku))
            continue;
        elseif(i > min(ku + 1 + kl, j + kl))
            continue;
        else
            bS(bRow, bCol) = -1 * bS(bRow, bCol);
        end
    end
end

% Scale matrix
bS = 0 * bS;

csvwrite('syn.csv', bS)

%% Run CUDA simulation
system('./dabrain');
load('firing.log');

figure;
plot(firing(:, 1), firing(:, 2), '.');
title('Neuron firing CUDA');