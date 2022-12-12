%%% test some fucking fro norms my dude%%
  
  %move the csv files from cpp into the same folder as matlab code
  %run matlab at equivalent N size
  %don't clear any of the variables
  %run froTest

mlX = output.X;
mlZ = output.Z;

cppX = readmatrix('Xout.csv');
cppZ = readmatrix('Zout.csv');

froDifX = norm(mlX - cppX, 'fro');
froExpectedX = norm(mlX, 'fro');

froDifZ = norm(mlZ - cppZ, 'fro');
froExpectedZ = norm(mlZ, 'fro');

errX = (froDifX / froExpectedX) * 100
errZ = (froDifZ / froExpectedZ) * 100

