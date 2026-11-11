# Fase 3 — Tarea B v2 (dataset_B2, 306 sims, 272 con contacto)
Control pose_only (acc 0.357 ± 0.068, azar 0.333): el atajo de diseño desapareció.
Forma (3 clases, 5-fold estratificado, 2 semillas): majority 0.346 | rf_contact_map 0.386 ± 0.080 | cnn_lstm 0.390 ± 0.049 | tcn 0.366 ± 0.045. Ningún modelo se separa del azar (0.333) ni del control pose_only: la forma NO es discriminable con este contacto.
Posición (x,y): majority 7.07 mm | rf_contact_map 1.78 ± 0.11 mm | cnn_lstm 3.15 ± 0.60 | tcn 3.90 ± 0.70. La posición sí se estima bien (RF mejor que redes con ~220 muestras de entrenamiento).
34/306 sims sin contacto útil (|M|<0.6) excluidas. Mediana |M| 1.3–1.7, 10–16 canales activos.
Conclusión: con cabeza quieta y solo toques de punta, la forma no se identifica. Para B como clasificación de forma hace falta contacto más rico (cabeza que se acerca, objetos más grandes/cercanos, barrido más amplio) o reformular B como localización de objeto + tamaño. Decisión con José Luis.
