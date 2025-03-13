% Carga de la red preentrenada GoogLeNet
net = googlenet;

% Verificación del tamaño de entrada de la red
inputSize = net.Layers(1).InputSize; % [224 224 3]

% Ruta a las imágenes (reemplázala con tus propias rutas)
imageFiles = {'./img/coches.jpg', './img/leon.png', './img/pimientos.jpg'}; % Especifica tus imágenes

% Aumentar el tamaño de las imágenes procesadas
displaySize = [448 448]; % Doble del tamaño original de GoogLeNet (224x224)

% Calcular disposición del grid
nImages = length(imageFiles);
nCols = ceil(sqrt(nImages));
nRows = ceil(nImages/nCols);

% Crear una figura más grande
figure('Name', 'Resultados de Clasificación con GoogLeNet', ...
       'NumberTitle', 'off', ...
       'Position', [50 50 1600 1000], ... % Ventana más grande
       'Color', [0.95 0.95 0.95]);

% Bucle para procesar y mostrar imágenes
for i = 1:length(imageFiles)
    % Carga y preparación de la imagen
    img = imread(imageFiles{i});
    % Redimensionar primero a tamaño de entrada de la red
    imgNetSize = imresize(img, [inputSize(1) inputSize(2)]);
    % Luego redimensionar al tamaño de visualización más grande
    imgResized = imresize(img, displaySize);
    
    % Convertir a RGB si es necesario
    if size(imgNetSize, 3) == 1
        imgNetSize = cat(3, imgNetSize, imgNetSize, imgNetSize);
        imgResized = cat(3, imgResized, imgResized, imgResized);
    end
    
    % Clasificación (usando el tamaño de entrada requerido por la red)
    [label, scores] = classify(net, imgNetSize);
    [topScores, topIdx] = maxk(scores, 5);
    topLabels = net.Layers(end).ClassNames(topIdx);
    
    % Crear subplot para cada imagen
    subplot(nRows, nCols, i);
    imshow(imgResized);
    
    % Crear título con formato (tamaño más grande)
    titleText = sprintf('%s (%.1f%%)', char(label), max(scores)*100);
    title(titleText, ...
          'FontSize', 12, ... % Título más grande
          'FontWeight', 'bold', ...
          'Interpreter', 'none');
    
    % Añadir información adicional como texto (tamaño ajustado)
    predText = sprintf('Top-2:\n%s (%.1f%%)\n%s (%.1f%%)', ...
        char(topLabels(2)), topScores(2)*100, ...
        char(topLabels(3)), topScores(3)*100);
    text(10, displaySize(1)-10, predText, ...
         'FontSize', 10, ... % Texto más grande
         'Color', 'white', ...
         'BackgroundColor', [0 0 0 0.5], ...
         'Interpreter', 'none');
    
    % Mejorar la apariencia
    set(gca, 'Box', 'off', ...
            'XTick', [], 'YTick', [], ...
            'LineWidth', 1);
end

% Ajustar el título general
sgtitle('Clasificaciones de Imágenes', ...
        'FontSize', 16, ... % Título general más grande
        'FontWeight', 'bold');
tight_layout();

% Función helper para ajustar el espaciado
function tight_layout()
    ax = findall(gcf, 'Type', 'axes');
    for i = 1:length(ax)
        outerpos = ax(i).OuterPosition;
        ti = ax(i).TightInset; 
        left = outerpos(1) + ti(1);
        bottom = outerpos(2) + ti(2);
        ax_width = outerpos(3) - ti(1) - ti(3);
        ax_height = outerpos(4) - ti(2) - ti(4);
        ax(i).Position = [left bottom ax_width ax_height];
    end
end