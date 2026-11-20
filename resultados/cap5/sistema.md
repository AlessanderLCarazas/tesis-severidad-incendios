# Metricas nuevas del sistema completo en la prueba (bloque del asesor, despues de la fase F)

Generado por `python metricas_sistema.py` sobre las salidas por fotograma de los modelos congelados (`runs_fuego/salidas_congeladas/`; reproducen el umbral conformal y el F1 de la fase F con diferencia 0.0e+00). Nada de esto se uso para elegir el sistema. Modelos: CfC, GRU y LSTM con 3 semillas (no hay modelos Mamba entrenados).

## B4. Confiabilidad de las probabilidades de nivel por segundo

Probabilidad de cada nivel por segundo: media de los fotogramas del segundo de P(controlado) = 1 - p1, P(incontrolado) = min(p1, p2) y P(incipiente) = p1 - P(incontrolado), las mismas de la fase F. Segundos dentro del tramo util con al menos un fotograma. ECE con 10 intervalos iguales de la confianza (probabilidad del nivel mas probable) y Brier multiclase (suma de los 3 niveles, de 0 a 2). Media +- desviacion de 3 semillas.

| modelo | referencia | segundos | ECE | Brier |
|---|---|---|---|---|
| CfC | IA | 4703 | 0.170 +- 0.016 | 0.644 +- 0.028 |
| CfC | humana | 4668 | 0.241 +- 0.023 | 0.752 +- 0.044 |
| GRU | IA | 4703 | 0.289 +- 0.008 | 0.716 +- 0.016 |
| GRU | humana | 4668 | 0.331 +- 0.017 | 0.817 +- 0.027 |
| LSTM | IA | 4703 | 0.221 +- 0.014 | 0.662 +- 0.014 |
| LSTM | humana | 4668 | 0.294 +- 0.025 | 0.789 +- 0.024 |

## B5. Cobertura de la alerta conformal (alfa 0.1)

Misma unidad que la calibracion: el video. Un video tiene falsa alarma si la alerta se activa en algun momento (equivale a que su riesgo maximo llegue al umbral). El umbral se calibro con los 52 videos de calibracion sin fuego o controlados, juntos; la garantia conformal es para esa poblacion junta (tasa esperada <= 10 %), no para cada grupo por separado. IC 95 % exacto de Clopper-Pearson por corrida; se cuenta en cuantas de las 3 corridas el intervalo incluye el 10 %.

| conjunto | modelo | controlados | sin fuego | juntos | corridas con el 10 % dentro del IC (juntos) |
|---|---|---|---|---|---|
| calibracion | CfC | 20.0 +- 0.0 % (5.0 de 25) | 0.0 +- 0.0 % (0.0 de 27) | 9.6 +- 0.0 % (5.0 de 52) | 3 de 3 |
| calibracion | GRU | 20.0 +- 0.0 % (5.0 de 25) | 0.0 +- 0.0 % (0.0 de 27) | 9.6 +- 0.0 % (5.0 de 52) | 3 de 3 |
| calibracion | LSTM | 20.0 +- 0.0 % (5.0 de 25) | 0.0 +- 0.0 % (0.0 de 27) | 9.6 +- 0.0 % (5.0 de 52) | 3 de 3 |
| validacion | CfC | 18.7 +- 3.8 % (4.7 de 25) | 0.0 +- 0.0 % (0.0 de 28) | 8.8 +- 1.8 % (4.7 de 53) | 3 de 3 |
| validacion | GRU | 14.7 +- 5.0 % (3.7 de 25) | 0.0 +- 0.0 % (0.0 de 28) | 6.9 +- 2.4 % (3.7 de 53) | 3 de 3 |
| validacion | LSTM | 9.3 +- 1.9 % (2.3 de 25) | 0.0 +- 0.0 % (0.0 de 28) | 4.4 +- 0.9 % (2.3 de 53) | 3 de 3 |
| prueba (IA) | CfC | 23.3 +- 7.2 % (7.0 de 30) | 0.0 +- 0.0 % (0.0 de 32) | 11.3 +- 3.5 % (7.0 de 62) | 3 de 3 |
| prueba (IA) | GRU | 22.2 +- 3.1 % (6.7 de 30) | 0.0 +- 0.0 % (0.0 de 32) | 10.8 +- 1.5 % (6.7 de 62) | 3 de 3 |
| prueba (IA) | LSTM | 23.3 +- 7.2 % (7.0 de 30) | 0.0 +- 0.0 % (0.0 de 32) | 11.3 +- 3.5 % (7.0 de 62) | 3 de 3 |
| prueba (humana) | CfC | 18.4 +- 6.5 % (5.3 de 29) | 0.0 +- 0.0 % (0.0 de 32) | 8.7 +- 3.1 % (5.3 de 61) | 3 de 3 |
| prueba (humana) | GRU | 20.7 +- 2.8 % (6.0 de 29) | 0.0 +- 0.0 % (0.0 de 32) | 9.8 +- 1.3 % (6.0 de 61) | 3 de 3 |
| prueba (humana) | LSTM | 23.0 +- 4.3 % (6.7 de 29) | 0.0 +- 0.0 % (0.0 de 32) | 10.9 +- 2.0 % (6.7 de 61) | 3 de 3 |

## B6. Estabilidad: cambios de nivel por minuto

Cambios del nivel por segundo dentro del tramo util, sobre el total de minutos. Referencia de la IA: los 70 videos de fuego; referencia humana: solo los 29 videos con cambios marcados reproduciendo el video (los otros 40 tienen un solo nivel por construccion).

| modelo | referencia | minutos | cambios por minuto del modelo (final) | sin histeresis | cambios por minuto de la referencia |
|---|---|---|---|---|---|
| CfC | IA | 78.4 | 2.10 +- 0.09 | 5.73 +- 1.13 | 0.41 |
| CfC | humana | 49.9 | 2.48 +- 0.11 | 6.23 +- 0.97 | 0.40 |
| GRU | IA | 78.4 | 1.59 +- 0.10 | 2.02 +- 0.24 | 0.41 |
| GRU | humana | 49.9 | 1.63 +- 0.06 | 2.30 +- 0.20 | 0.40 |
| LSTM | IA | 78.4 | 2.53 +- 0.21 | 4.43 +- 0.61 | 0.41 |
| LSTM | humana | 49.9 | 2.81 +- 0.12 | 4.98 +- 0.65 | 0.40 |

## B7. Costo de cada modelo temporal

Un paso = un fotograma con lote 1 (compuerta, celda o paso recurrente y las tres cabezas), como en el sistema en vivo, sin la normalizacion. 200 pasos de calentamiento y 2000 medidos; mediana y percentil 95. CPU con 8 hilos de PyTorch; GPU NVIDIA GeForce RTX 3050 Ti Laptop GPU con sincronizacion en cada paso. Los parametros son los mismos en las 3 semillas.

| modelo | parametros | CPU, mediana (ms) | CPU, p95 (ms) | GPU, mediana (ms) | GPU, p95 (ms) |
|---|---|---|---|---|---|
| CfC | 46133 | 0.185 | 0.205 | 0.706 | 0.849 |
| GRU | 19829 | 0.148 | 0.173 | 0.403 | 0.488 |
| LSTM | 26293 | 0.246 | 0.325 | 0.430 | 0.558 |

## B8. Error absoluto del tiempo al peligro

Solo en los fotogramas anteriores al cambio a incontrolado de los videos donde la referencia lo define (tau_mascara = 1, tau > 0). El MAE reproduce el de la fase F (diferencia 0.0e+00). Constante: la mediana de tau en entrenamiento (52.7 s).

| modelo | referencia | videos | fotogramas | error absoluto mediano (s) | MAE (s) |
|---|---|---|---|---|---|
| CfC | IA | 21 | 5802 | 15.1 +- 1.0 | 18.9 +- 1.2 |
| CfC | humana | 16 | 10293 | 45.4 +- 3.5 | 63.4 +- 2.8 |
| GRU | IA | 21 | 5802 | 19.4 +- 2.1 | 25.7 +- 2.3 |
| GRU | humana | 16 | 10293 | 41.2 +- 0.6 | 58.4 +- 0.7 |
| LSTM | IA | 21 | 5802 | 16.7 +- 2.6 | 20.1 +- 1.9 |
| LSTM | humana | 16 | 10293 | 42.3 +- 2.0 | 60.1 +- 2.3 |
| Constante (mediana de entrenamiento) | IA | 21 | 5802 | 33.3 +- 0.0 | 30.4 +- 0.0 |
| Constante (mediana de entrenamiento) | humana | 16 | 10293 | 36.8 +- 0.0 | 48.1 +- 0.0 |

