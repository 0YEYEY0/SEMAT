% Cargar los datos desde un archivo Excel
filename = 'datosT.xlsx';
data = readtable(filename);

% Separar las variables de entrada (X) y la variable objetivo (Y)
X = data{:, 1:end-1}; % Todas las columnas excepto la última
Y = data{:, end};     % Última columna (Temperatura interna)

% Normalizar los datos
[X_norm, mu, sigma] = zscore(X);
[Y_norm, muY, sigmaY] = zscore(Y);

#Dividir los datos en entrenamiento (70%) y prueba (30%)
cv = cvpartition(size(X_norm,1), 'HoldOut', 0.3);
X_train = X_norm(training(cv), :);
Y_train = Y_norm(training(cv), :);
X_test  = X_norm(test(cv), :);
Y_test  = Y_norm(test(cv), :);

% Crear la red neuronal
hiddenLayerSize = 10; % Número de neuronas en la capa oculta
net = feedforwardnet(hiddenLayerSize);

% Configurar los datos de entrenamiento
net = configure(net, X_train', Y_train');

% Entrenar la red neuronal
net = train(net, X_train', Y_train');

% Realizar predicciones
Y_pred_norm = net(X_test');
Y_pred = Y_pred_norm' * sigmaY + muY; % Desnormalizar la salida

% Calcular error
error = mean(abs(Y_pred - (Y_test * sigmaY + muY)));
disp(['Error medio absoluto: ', num2str(error), ' °C']);

% Graficar resultados
figure;
plot(Y_test * sigmaY + muY, 'b', 'LineWidth', 1.5); hold on;
plot(Y_pred, 'r--', 'LineWidth', 1.5);
legend('Temperatura real', 'Temperatura predicha');
xlabel('Muestras de prueba');
ylabel('Temperatura interna (°C)');
title('Comparación entre temperatura real y predicha');
grid on;







% 1. Cargar los datos desde el archivo Excel
nombreArchivo = 'datos.xlsx'; % Nombre del archivo

% Verificar que el archivo existe
if ~isfile(nombreArchivo)
    error('El archivo "%s" no se encuentra en el directorio actual.', nombreArchivo);
end

% Leer datos del archivo Excel
datos = readmatrix(nombreArchivo);

% 2. Separar entradas (X) y salida (Y)
X = datos(:, 1:end-1)';  % Entradas (transpuesta para formato MATLAB)
Y = datos(:, end)';      % Salida esperada (columna "Desbalance")

% Verificar si hay datos vacíos o no numéricos
if any(isnan(X(:))) || any(isnan(Y(:)))
    error('El archivo contiene valores NaN o no numéricos. Verifique el archivo.');
end

% Normalizar los datos para mejorar el rendimiento de la red
[X, X_settings] = mapminmax(X);
[Y, Y_settings] = mapminmax(Y);

% 3. Dividir los datos en entrenamiento (70%), validación (15%) y prueba (15%)
numMuestras = size(X, 2);
idx = randperm(numMuestras); % Mezclar índices aleatoriamente

trainIdx = idx(1:round(0.7 * numMuestras));
valIdx = idx(round(0.7 * numMuestras) + 1:round(0.85 * numMuestras));
testIdx = idx(round(0.85 * numMuestras) + 1:end);

X_train = X(:, trainIdx);  Y_train = Y(:, trainIdx);
X_val   = X(:, valIdx);    Y_val   = Y(:, valIdx);
X_test  = X(:, testIdx);   Y_test  = Y(:, testIdx);

% 4. Crear la red neuronal con una capa oculta de 10 neuronas
hiddenLayerSize = 10;
net = feedforwardnet(hiddenLayerSize);

% 5. Configurar la división de los datos manualmente
net.divideFcn = 'divideind';
net.divideParam.trainInd = trainIdx;
net.divideParam.valInd = valIdx;
net.divideParam.testInd = testIdx;

% Configurar parámetros de entrenamiento
net.trainParam.epochs = 1000; % Número de épocas
net.trainParam.goal = 1e-5;   % Tolerancia mínima de error
net.trainParam.min_grad = 1e-6; % Gradiente mínimo para detener el entrenamiento

% 6. Entrenar la red neuronal
net = train(net, X, Y);

% 7. Evaluar el desempeño con los datos de prueba
Y_pred = net(X_test); % Salida de la red neuronal

% Desnormalizar los datos para comparar con los valores reales
Y_pred = mapminmax('reverse', Y_pred, Y_settings);
Y_test = mapminmax('reverse', Y_test, Y_settings);

% Calcular el error de la red neuronal
performance = perform(net, Y_test, Y_pred);

% 8. Mostrar resultados
disp('Error de la red neuronal:');
disp(performance);

% 9. Graficar resultados reales vs predichos
figure;
plot(Y_test, 'b', 'LineWidth', 2); hold on;
plot(Y_pred, 'r--', 'LineWidth', 2);
legend('Valores Reales', 'Valores Predichos', 'Location', 'Best');
xlabel('Muestras');
ylabel('Desbalance');
title('Comparación de la Salida de la Red Neuronal');
grid on;
