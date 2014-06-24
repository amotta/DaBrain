% Created by Eugene M. Izhikevich, February 25, 2003
% Available for download on the author's website:
% http://www.izhikevich.org/publications/spikes.htm
%
% Modified by Alessandro Motta and Hugo Zeberg

clear all
rng(0,'CombRecursive')
fVect = [0.001, 0.002, 0.003, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 1, 2, 3];
numRep = 5;
timings = zeros(length(fVect), numRep);

for fIndex = 1:length(fVect)
    f = fVect(fIndex)
    for curRep = 1:numRep
        clearvars -except f fIndex curRep numRep fVect timings

        % Excitatory neurons            Inhibitory neurons
        % Number of different neurons
        Ne = round(800 * f);            Ni = round(200 * f);
        % Random number to differentiate neurons
        re = rand(Ne, 1);               ri = rand(Ni, 1);

        % Total number of neurons
        N = Ne + Ni;

        % Neuronal parameters
        a = [0.02 * ones(Ne, 1);        0.02 + 0.08 * ri];
        b = [0.2 * ones(Ne, 1);         0.25 - 0.05 * ri];
        % Membrane voltage reset
        c = [-65 + 15 * re .^ 2;        -65 * ones(Ni, 1)];
        % Recovery variable reset
        d = [8 - 6 * re .^ 2;           2 * ones(Ni, 1)];

        % Synapse matrix
        S = [0.5 * rand(N, Ne),         -rand(N, Ni)];

        % Initial membrane voltage
        v = -65 * ones(N, 1);
        % Initial recovery value
        u = b .* v;

        % Simulation
        I = zeros(N, 1);
        Isyn = zeros(N, 1);
        Ithal = zeros(N, 1);
        fired = zeros(N, 1);

        tEnd = 1000;
        firings = zeros(tEnd, N);

        tic
        for t = 1 : tEnd
            % find firing neurons
            firedId = find(v >= 30);

            fired = zeros(N, 1);
            fired(firedId) = 1;
            firings(t, : ) = fired;

            % reset firing neurons
            v(firedId) = c(firedId);
            u(firedId) = u(firedId) + d(firedId);

            % compute current
            Isyn = S * fired;
            Ithal = 1.1 * [5 * randn(Ne, 1); 2 * randn(Ni, 1)];
            I = Isyn + Ithal;

            % update state
            v = v + 0.5 * (0.04 * v .^ 2 + 5 * v + 140 - u + I);
            v = v + 0.5 * (0.04 * v .^ 2 + 5 * v + 140 - u + I);
            u = u + a .* (b .* v - u);
        end
        timings(fIndex, curRep) = toc;
    end
end
%%
% Visualization
figure;
[firingTime, firingId] = find(firings);
plot(firingTime, firingId, '.');

figure;
plot(1 : tEnd, sum(firings, 2))

%%
% Spectral analysis
Fs = 1000;
f = Fs / 2 * linspace(0, 1, tEnd / 2 + 1);
y = fft(sum(firings, 2));

figure;
plot(f, 2 * abs(y(1 : (tEnd / 2 + 1))));
axis([0, 60, 0, 10000]);
