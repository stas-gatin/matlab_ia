load('modelo_entrenado.mat', 'net');
YPred = classify(net, imdsValidation);
YValidation = imdsValidation.Labels;

accuracy = sum(YPred == YValidation) / numel(YValidation);
disp("Precisión: " + accuracy);