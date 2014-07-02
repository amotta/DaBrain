clear all;
load('current.log');

% data agnostic
data = current;

% Configuration
% Sampling frequency
Fs = 200;
% Sampling interval
T = 1 / Fs;
% Length of signal (in # of samples)
L = 4 * Fs;
% Time vector
t = (0 : L-1) * T;
% For radix-2 FFT
NFFT = 2 ^ nextpow2(L);
% Frequency vector
f = Fs / 2 * linspace(0, 1, NFFT / 2 + 1);

% Sum over all neurons
y = sum(data(200:end, 2:end), 2);
% Eliminate DC signal
y = y - mean(y);
% .. and normalize signal
y = y / max(abs(y));
% Perform FFT
Y = fft(y, NFFT)/L;

figure;
hold all;
plot( ...
    f, ...
    smooth((2 * abs(Y(1:NFFT/2+1)) .^ 2), 10, 'moving') ...
);

% Layout
title('Power spectrum of network of hippocampal neurons');
xlabel('Frequency (Hz)');
ylabel('P(f)');