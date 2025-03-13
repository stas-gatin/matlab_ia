% Carga de la red preentrenada GoogLeNet
net = googlenet;

% Verificación del tamaño de entrada de la red (GoogLeNet requiere imágenes de 224x224)
inputSize = net.Layers(1).InputSize; % Normalmente [224 224 3]

% Ruta a las imágenes (reemplázala con tus propias rutas)
imageFiles = {'./img/coches.jpg'}; % Especifica tus imágenes

% Bucle para procesar varias imágenes
for i = 1:length(imageFiles)
    % Carga de la imagen
    img = imread(imageFiles{i});
    
    % Preparación de la imagen: cambio de tamaño y normalización
    imgResized = imresize(img, [inputSize(1) inputSize(2)]);
    
    % Si la imagen es en blanco y negro, la convertimos a RGB
    if size(imgResized, 3) == 1
        imgResized = cat(3, imgResized, imgResized, imgResized);
    end
    
    % Clasificación de la imagen
    [label, scores] = classify(net, imgResized);
    
    % Obtención de las 5 mejores predicciones (opcional)
    [topScores, topIdx] = maxk(scores, 5); % 5 mejores probabilidades
    topLabels = net.Layers(end).ClassNames(topIdx); % Etiquetas de las 5 mejores predicciones
    
    % Mostrar el resultado
    fprintf('Imagen %d: %s\n', i, imageFiles{i});
    fprintf('Etiqueta predicha: %s (probabilidad %.2f%%)\n', char(label), max(scores)*100);
    fprintf('Top-5 predicciones:\n');
    for j = 1:5
        fprintf('  %s: %.2f%%\n', char(topLabels(j)), topScores(j)*100);
    end
    fprintf('\n');
    
    % Mostrar la imagen con la etiqueta predicha
    figure(i); % Abre una nueva ventana para cada imagen
    imshow(imgResized);
    title(sprintf('Predicción: %s (%.2f%%)', char(label), max(scores)*100));
end
