% Generating all CSV files

% Thousands of neurons
f = 0.1; 
% Proportion of inhibitory neurons
pInh = 0.0;
% Upper and lower diagonals
ku = 10;
kl = 10;

N = round(f * 1000);
Ni = round(pInh * N);
Ne = N - Ni;

% Index of inhibitory neurons
posI = round(linspace(1, N, Ni));

% Random number to differentiate neurons
r = rand(N, 1);
ri = rand(Ni, 1);

% K conductance
gK = 36 * (0.95 + 0.1 * r);
% Na conductance
gNa = 120 * (0.95 + 0.1 * r);
% Leakage
gL = 0.3 * (0.95 + 0.1 * r);

% Write data to CSV files
dynParam = [gK'; gNa'; gL'];
csvwrite('dynParam.csv', dynParam);

% Initial membrane voltage
vZero = zeros(N, 1);

% K activation
nAlphaZero = 0.01 * 10 / (exp(10 / 10) - 1);
nBetaZero = 0.125;
nZero = nAlphaZero / (nAlphaZero + nBetaZero) * ones(N, 1);

% Na activation
mAlphaZero = 0.1 * 25 / (exp(25 / 10) - 1);
mBetaZero = 4;
mZero = mAlphaZero / (mAlphaZero + mBetaZero) * ones(N, 1);

% Na inactivation
hAlphaZero = 0.07;
hBetaZero = 1 / (exp(30 / 10) + 1);
hZero = hAlphaZero / (hAlphaZero + hBetaZero) * ones(N, 1);

% Save initial states
dynState = [vZero'; nZero'; mZero'; hZero'];
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
    i
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

csvwrite('syn.csv', bS)

%% Run CUDA simulation
system('./dabrain');
load('firing.log');

figure;
plot(firing(:, 1), firing(:, 2), '.');
title('Neuron firing CUDA');