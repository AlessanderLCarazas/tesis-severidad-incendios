# Variantes de la GRU de escena en validacion

Generado desde `resultados/variantes_gru/*.json` (`python entrenar_variantes.py --variante <v> --semilla <s>`). Semillas 42, 43 y 44; media +- desviacion. La prueba no se usa.

| variante | F1 macro antes | F1 macro con histeresis | F1 incipiente antes | F1 incipiente con histeresis | kappa antes | kappa con histeresis | cambios por video con histeresis |
|---|---|---|---|---|---|---|---|
| base (CORAL, tres perdidas) | 0.621 +- 0.008 | 0.612 +- 0.002 | 0.208 +- 0.052 | 0.209 +- 0.040 | 0.698 +- 0.022 | 0.679 +- 0.026 | 2.21 |
| a. softmax con entropia cruzada | 0.623 +- 0.059 | 0.611 +- 0.057 | 0.365 +- 0.060 | 0.346 +- 0.060 | 0.605 +- 0.083 | 0.592 +- 0.080 | 1.79 |
| b. doble peso para el incipiente | 0.630 +- 0.022 | 0.615 +- 0.021 | 0.317 +- 0.073 | 0.303 +- 0.070 | 0.652 +- 0.028 | 0.633 +- 0.023 | 2.45 |
| c. solo la perdida de nivel | 0.609 +- 0.023 | 0.604 +- 0.018 | 0.225 +- 0.054 | 0.239 +- 0.042 | 0.658 +- 0.017 | 0.638 +- 0.016 | 2.12 |

| variante | semilla | mejor epoca | F1 macro | F1 incipiente | kappa | segundos |
|---|---|---|---|---|---|---|
| base (CORAL, tres perdidas) | 42 | 58 | 0.630 | 0.274 | 0.668 | 69 |
| base (CORAL, tres perdidas) | 43 | 57 | 0.610 | 0.148 | 0.720 | 72 |
| base (CORAL, tres perdidas) | 44 | 24 | 0.623 | 0.204 | 0.706 | 76 |
| a. softmax con entropia cruzada | 42 | 24 | 0.560 | 0.293 | 0.523 | 73 |
| a. softmax con entropia cruzada | 43 | 46 | 0.701 | 0.440 | 0.718 | 42 |
| a. softmax con entropia cruzada | 44 | 42 | 0.607 | 0.361 | 0.573 | 48 |
| b. doble peso para el incipiente | 42 | 41 | 0.661 | 0.392 | 0.658 | 46 |
| b. doble peso para el incipiente | 43 | 24 | 0.612 | 0.340 | 0.616 | 46 |
| b. doble peso para el incipiente | 44 | 26 | 0.616 | 0.219 | 0.683 | 44 |
| c. solo la perdida de nivel | 42 | 54 | 0.622 | 0.244 | 0.677 | 40 |
| c. solo la perdida de nivel | 43 | 47 | 0.577 | 0.151 | 0.636 | 49 |
| c. solo la perdida de nivel | 44 | 54 | 0.628 | 0.279 | 0.662 | 48 |
