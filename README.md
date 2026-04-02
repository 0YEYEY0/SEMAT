# SEMAT

El proyecto entrena un autoencoder con ventanas temporales de motores sanos y luego evalua archivos nuevos para clasificar cada motor como `SANO` o `FALLA`.

## Resumen

El flujo actual hace esto:

- Lee archivos `.xlsx` con mediciones del motor.
- Soporta hojas con nombre `Sheet1` o una hoja unica con otro nombre.
- Soporta dos formatos de encabezado vistos en los datos.
- Ordena cada archivo por tiempo usando `Muestra` o `Etiqueta_de_tiempo`.
- Construye ventanas temporales consecutivas.
- Entrena un autoencoder sobre motores sanos.
- Calcula un umbral de anomalia.
- Evalua archivos nuevos y emite un veredicto final.

## Estructura del proyecto

```text
SEMAT/
├── config.json
├── requirements.txt
├── train_autoencoder.py
├── evaluate_motor.py
├── src/
│   ├── dataset_builder.py
│   ├── excel_utils.py
│   ├── model.py
│   └── training_utils.py
├── data/
│   ├── sanos/
│   │   ├── carga_33/
│   │   ├── carga_50/
│   │   ├── carga_75/
│   │   └── carga_100/
│   ├── por_evaluar/
│   │   ├── carga_33/
│   │   ├── carga_50/
│   │   ├── carga_75/
│   │   └── carga_100/
│   ├── detectado_sano/
│   └── detectado_falla/
└── models/
    ├── autoencoder_motor.pt
    ├── scaler.pkl
    ├── umbral.json
    ├── history.json
    └── split_info.json
```

## Requisitos

- Python 3.11 recomendado
- Windows PowerShell

Dependencias principales:

- `pandas`
- `numpy`
- `scikit-learn`
- `joblib`
- `openpyxl`
- `torch`

## Instalacion

Crear entorno virtual e instalar dependencias:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Datos de entrada

### Datos sanos para entrenamiento

Los archivos sanos deben colocarse en:

- `data/sanos/carga_33`
- `data/sanos/carga_50`
- `data/sanos/carga_75`
- `data/sanos/carga_100`

La carga se infiere desde el nombre de la carpeta:

- `carga_33 -> 0.33`
- `carga_50 -> 0.50`
- `carga_75 -> 0.75`
- `carga_100 -> 1.00`

### Archivos para evaluacion

Los archivos nuevos deben colocarse en:

- `data/por_evaluar/carga_33`
- `data/por_evaluar/carga_50`
- `data/por_evaluar/carga_75`
- `data/por_evaluar/carga_100`

Esto es importante porque la carga forma parte de las features del modelo.

## Entrenamiento

Para entrenar el autoencoder con los archivos sanos:

```powershell
.\venv\Scripts\python.exe .\train_autoencoder.py
```

El script:

- carga todos los archivos sanos,
- elimina filas incompletas,
- divide los archivos por carga en train y validation,
- normaliza las features,
- construye ventanas temporales,
- entrena el autoencoder,
- guarda el modelo y artefactos en `models/`.

### Archivos generados en `models/`

- `autoencoder_motor.pt`: pesos del modelo entrenado
- `scaler.pkl`: media y desviacion usadas para normalizar
- `umbral.json`: umbral de error de reconstruccion por ventana
- `history.json`: historial de `train_loss` y `val_loss`
- `split_info.json`: archivos usados en train y validation

## Evaluacion de un archivo nuevo

Para evaluar un archivo:

```powershell
.\venv\Scripts\python.exe .\evaluate_motor.py .\data\por_evaluar\carga_75\archivo.xlsx
```

Salida esperada:

- `Error medio`
- `Fraccion anomala`
- `Umbral fraccion archivo`
- `Ventanas evaluadas`
- `Veredicto: SANO` o `Veredicto: FALLA`



segun el resultado final.

## Configuracion

Los parametros principales estan en [`config.json`](/SEMAT/config.json):

- `window_size`: tamano de cada ventana temporal
- `window_stride`: salto entre ventanas
- `feature_columns`: variables usadas por el modelo
- `include_load_column`: incluye la carga como feature
- `file_anomaly_fraction_threshold`: porcentaje de ventanas anomalas necesario para marcar un archivo como falla
- `threshold_percentile`: percentil usado para construir el umbral de anomalia por ventana

Valores actuales relevantes:

```json
{
  "window_size": 64,
  "window_stride": 16,
  "file_anomaly_fraction_threshold": 0.30,
  "threshold_percentile": 99.0
}
```

## Como funciona la decision

1. El modelo reconstruye cada ventana temporal del archivo.
2. Se calcula el error de reconstruccion de cada ventana.
3. Una ventana se considera anomala si supera el umbral guardado en `models/umbral.json`.
4. Si la fraccion de ventanas anomalas supera `file_anomaly_fraction_threshold`, el archivo completo se clasifica como `FALLA`.

## Ejemplos

Entrenar:

```powershell
.\venv\Scripts\python.exe .\train_autoencoder.py
```

Evaluar un sano:

```powershell
.\venv\Scripts\python.exe .\evaluate_motor.py .\data\por_evaluar\carga_75\E01_B01_L75_DBnorm_R2.xlsx
```

Evaluar una falla:

```powershell
.\venv\Scripts\python.exe .\evaluate_motor.py .\data\por_evaluar\carga_33\E13_B02_L30_DB4%_R1.xlsx
```

## Limitaciones actuales

- El modelo depende del formato de las columnas esperadas en los Excel.
- La carga se infiere por carpeta, no por metadato interno del archivo.
- El modelo fue entrenado con un conjunto pequeno de archivos sanos.
- La calidad final del clasificador depende de validar con mas motores sanos y con falla real.



## Estado actual

El proyecto ya permite:

- entrenar un modelo con motores sanos,
- evaluar archivos nuevos,
- usar ventanas temporales para capturar arranque y estabilizacion,
- incluir la carga del motor en la evaluacion.

