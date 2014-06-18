% Generate square matrix with random entries
% and save it in a CSV file.
matSize = 100;
mat = rand(matSize);

csvwrite('randSquare.csv', mat)