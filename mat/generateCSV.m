% Generating all CSV files

% Thousands of neurons
f = 1; 
% Proportion of inhibitory neurons
pInh = 0.0;
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

% Neuronal parameters
a = 0.02 * ones(N, 1);
a(posI) = 0.02 + 0.08 * ri;

b = 0.2 * ones(N, 1);
b(posI) = 0.25 - 0.05 * ri;

% Membrane voltage reset
c = -65 + 15 * r .^ 2;
c(posI) = -65 * ones(Ni, 1);

% Recovery variable reset
d = 8 - 6 * r .^ 2;
d(posI) = 2 * ones(Ni, 1);

% Write data to CSV files
dynParam = [a'; b'; c'; d'];
csvwrite('dynParam.csv', dynParam);

% Initial membrane voltage
v = -65 * ones(N, 1);
% Initial recovery value
u = b .* v;

dynState = [v'; u'];
csvwrite('dynState.csv', dynState);

% banded synapse matrix
bS = zeros(ku + kl + 1, N);

% upper diagonals
for i = 1 : ku
    i
	piv = (ku + 1) - i;
	pre = zeros(1, piv);
	suf = (i / ku ) * rand(1, N - piv);
	bS(i, : ) = [pre, suf];
end

% lower diagonals
for i = (ku + 2) : (ku + kl + 1)
    i
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