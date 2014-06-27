% Generating all CSV files

% Thousands of neurons
f = 1;
% Proportion of inhibitory neurons
pInh = 0.2;
% Upper and lower diagonals
ku = 100;
kl = 100;

N = round(f * 1000);
Ni = round(pInh * N);
Ne = N - Ni;

% Index of inhibitory neurons
posI = round(linspace(1, N, Ni));

% Random number to differentiate neurons
r = rand(N, 1);
ri = rand(Ni, 1);

% Leakage
gL = 1 / 4.3E9 * (0.9 + 0.2 * r);

% Na channel density
pNa = 40E-6 * r;

% K channel density
pK = 20E-6 * r;

% Write data to CSV files
dynParam = [gL'; pNa'; pK'];
csvwrite('dynParam.csv', dynParam);

% Initial membrane voltage
v0 = -70E-3;
vZero = v0 * ones(N, 1);

% Na activation
mAlphaZero = 60000 * (vZero + 0.033) ./ (1 - exp(-(vZero + 0.033) / 0.003));
mBetaZero = -70000 * (vZero + 0.042) ./ (1 - exp((vZero + 0.042) / 0.02));
mZero = mAlphaZero ./ (mAlphaZero + mBetaZero);

% Na inactivation
hAlphaZero = -50000 * (vZero + 0.065) ./ (1 - exp((vZero + 0.065) / 0.006));
hBetaZero = 2250 ./ (1 + exp(-(vZero + 0.01) / 0.01));
hZero = hAlphaZero ./ (hAlphaZero + hBetaZero);

% K activation
nAlphaZero = 16000 * (vZero + 0.01) ./ (1 - exp(-(vZero + 0.01) / 0.01));
nBetaZero = -40000 * (vZero + 0.035) ./ (1 - exp((vZero + 0.035) / 0.01));
nZero = nAlphaZero ./ (nAlphaZero + nBetaZero);

% Save initial states
dynState = [vZero'; mZero'; hZero'; nZero'];
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
bS = 5E-12 * bS;

csvwrite('syn.csv', bS)

%% Run CUDA simulation
system('./dabrain');
load('firing.log');

figure;
plot(firing(:, 1), firing(:, 2), '.');
title('Neuron firing CUDA');