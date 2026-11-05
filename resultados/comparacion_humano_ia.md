## calidad de las etiquetas humanas
videos: 40
segundos entre guardados (39 intervalos): minimo 1, mediana 4, media 6, maximo 49
intervalos menores a 10 s: 35, menores a 20 s: 37
tiempo total de la sesion: 3.8 min
sin nivel inicial (no descartados): 0
descartados: 0
sin inicio o fin util marcado (se toma el video completo): 40
cambios marcados: t_incipiente 0, t_incontrolado 0

## comparacion humano contra ia
videos en comun: 40, comparables (ambos con nivel y sin descartar): 40

### nivel inicial
acuerdo: 85.0 % (34 de 40)
kappa ponderado lineal: 0.801 | kappa ponderado cuadratico: 0.880 | kappa sin ponderar: 0.704
matriz de confusion (filas: humano, columnas: ia)
| humano \ ia | controlado | incipiente | incontrolado |
|---|---|---|---|
| controlado | 25 | 0 | 0 |
| incipiente | 3 | 3 | 2 |
| incontrolado | 0 | 1 | 6 |

### nivel maximo del video
acuerdo: 82.5 % (33 de 40)
kappa ponderado lineal: 0.770 | kappa ponderado cuadratico: 0.860 | kappa sin ponderar: 0.662
matriz de confusion (filas: humano, columnas: ia)
| humano \ ia | controlado | incipiente | incontrolado |
|---|---|---|---|
| controlado | 24 | 1 | 0 |
| incipiente | 3 | 3 | 2 |
| incontrolado | 0 | 1 | 6 |

### cambios de nivel
t_incipiente: en ambos 0, solo humano 0, solo ia 1
t_incontrolado: en ambos 0, solo humano 0, solo ia 0

| video | cambio | humano (s) | ia (s) |
|---|---|---|---|
| 12399227_3840_2160_60fps.mp4 | t_incipiente | - | 8.29 |

### videos con distinto nivel inicial
| video | humano | ia | motivo de la ia |
|---|---|---|---|
| 12858041_3840_2160_24fps.mp4 | incipiente | incontrolado | Varios focos de fuego en vegetacion y troncos del suelo del bosque con humo; superficie extensa. |
| 13229149_3840_2160_30fps.mp4 | incontrolado | incipiente | Foco pequeno y lejano de fuego en bosque con columna de humo; un solo foco de superficie reducida. |
| 14296810_3840_2160_24fps.mp4 | incipiente | incontrolado | Quema de campo extensa con frente de fuego amplio y mucho humo, superficie grande de vegetacion afectada. |
| 18642729-uhd_3840_2160_60fps.mp4 | incipiente | controlado | Primer plano de fogata con lena, llamas en su fuente y estables; solo parece grande por la cercania. |
| 1912650782.mp4 | incipiente | controlado | Fogata de lena con brasas, llamas contenidas en su fuente y de tamano estable. |
| 2354874365.mp4 | incipiente | controlado | Fogata pequena en la orilla de un lago con una columna de humo, tamano estable y sin propagacion. |
