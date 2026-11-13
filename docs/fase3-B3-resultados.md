# Fase 3 — Tarea B v3 (cabeza avanzando, dataset_B3 completo) — 2026-11-12
Datos: 306 sims (102 por clase), sin NaN. max|M| mediana 2.68 (B2: 1.5), p90 6.06, máx 14.0; 31 sims con max|M|<0.6 excluidas -> 275. Bigotes con |M|>0.3: mediana 7. Poses x 45–71.5 mm. Control pose_only previo: 0.265.
Protocolo: train_B.py, 5-fold estratificado, 2 semillas, 500 Hz.
| Modelo | Exactitud forma (azar 0.333) | F1 | Error posición (mm) |
| majority | 0.342 | 0.170 | 11.15 ± 0.79 |
| pose_only (control) | 0.360 ± 0.062 | 0.359 | — |
| rf_contact_map | 0.360 ± 0.042 | 0.353 | 2.56 ± 0.17 |
| cnn_lstm | 0.376 ± 0.044 | 0.348 | 5.11 ± 0.57 |
| tcn | 0.378 ± 0.053 | 0.362 | 5.88 ± 0.73 |
Diagnósticos (ExtraTrees, rasgos ricos: max, media, std, min, instante del pico, derivada): todas 0.391 ± 0.051; contacto >=2: 0.398 ± 0.063 (n=196); contacto >=3.5: 0.439 ± 0.097 (n=93); cilindro vs onda con contacto >=2: 0.609 ± 0.120 (azar 0.5, n=131).
Veredicto: la forma NO se clasifica de forma robusta ni con contacto más rico. Indicio débil de que mejora con contacto fuerte, dentro del ruido. Según el criterio de José Luis (12-nov): el paper queda en la base B (Tarea A + localización de objeto + hallazgo de no discriminabilidad de forma, ahora con dos regímenes de contacto).
Posición: RF 2.56 mm frente a 11.15 mm de la media (rango de poses más amplio que en B2).
