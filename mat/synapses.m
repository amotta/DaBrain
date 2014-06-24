% authors:
% Alessandro Motta, Hugo Zeberg
%
% generate a banded synapse matrix

% number of neurons
n = 1000;
% number of upper diagonals
ku = 50;
% number of lower diagonals
kl = 50;

% banded synapse matrix
bS = zeros(ku + kl + 1, n);

% upper diagonals
for i = 1:ku
	piv = (ku + 1) - i;
	pre = zeros(1, piv);
	suf = (i / ku ) * rand(1, n - piv);
	bS(i, : ) = [pre, suf];
end

% lower diagonals
for i = (ku + 2):(ku + kl + 1)
	piv = i - (ku + 1);
	pre = (kl + 1 - piv) / kl * rand(1, n - piv);
	suf = zeros(1, piv);
	bS(i, : ) = [pre, suf];
end

csvwrite('syn.csv',bS)
