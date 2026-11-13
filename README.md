# Localización de objetos y límites de la discriminación de forma con torques simulados de bigotes (WHISKiT)

Código, cuadernos, resultados y manuscrito del estudio "What Can Be Recovered from Whisker-Base Torque?" (Carolina Londoño Idárraga y Dr.-Ing. José Luis Zárate Moya, Universitat Internacional Valenciana).

Los momentos y fuerzas en la base de los 27 bigotes derechos de la rata se generan con [WHISKiT Physics](https://github.com/SeNSE-lab/whiskitphysics) (laboratorio SeNSE) y se usan para entrenar modelos que estiman la posición y la forma de un objeto.

## Resultados principales

| Experimento | Pregunta | Resultado |
|---|---|---|
| P (poste) | ¿Dónde está el poste? | Error de 1,76 mm (CNN-LSTM), 1,71 mm (Random Forest), 1,44 mm (TCN). La CNN-LSTM es robusta al ruido (1,84 mm con ruido del 20 %; Random Forest 5,65 mm) |
| D1 (forma acoplada a la pose) | ¿Qué forma tiene el objeto? | Exactitud aparente de 0,70, pero un control que solo ve la pose alcanza 0,62: atajo de diseño |
| D2 (forma desacoplada, cabeza quieta) | Igual | Forma al azar (0,39; azar 0,33). Posición: 1,78 mm |
| D3 (forma desacoplada, cabeza avanzando) | Igual | Forma al azar (0,38). Posición: 2,56 mm |

El manuscrito está en `paper/` (inglés y español).

## Estructura

- `dataset/`: generadores de simulaciones (`run_dataset*.py`), empaquetado (`build_npz*.py`), entrenamiento (`train_A.py`, `train_B.py`), robustez y ablación.
- `patches/whiskit_patches.diff`: cambios a WHISKiT Physics (posición, rotación y escala del objeto; cabeza que avanza; opciones de línea de comandos).
- `notebooks/`: cuadernos de Google Colab que compilan WHISKiT y generan cada conjunto de datos (sin GPU; es CPU).
- `results/`: resultados (JSON) y registros de los experimentos.
- `paper/`: manuscrito en LaTeX (IEEEtran), figuras y el programa que las genera.
- `setup_whiskit.sh`: clona WHISKiT, aplica los parches y compila.

## Cómo reproducir

1. `./setup_whiskit.sh` (Linux/Colab). Crea `whiskitphysics/code/build/whiskit`.
2. Generar datos, por ejemplo la Tarea A: `cd dataset && python3 run_dataset.py ...`, luego `python3 build_npz.py --dir <carpeta> --out dataset_A.npz`.
3. Entrenar: `python3 train_A.py --data dataset_A.npz ...` o `python3 train_B.py --data dataset_B2.npz`.

Los conjuntos `.npz` (cientos de MB) no se versionan; se regeneran con los cuadernos.

## Protocolo experimental

- Una muestra es una simulación completa (tiempo × 27 bigotes × 6 canales: Mx, My, Mz, Fx, Fy, Fz).
- Partición por simulación (nunca por ventana), validación cruzada (estratificada en B), 20 % de validación interna para parada temprana.
- Normalización ajustada solo con entrenamiento; varias semillas; errores en mm.
- Modelos base: media, Ridge, Random Forest sobre el mapa de contacto, y un control que solo ve la pose (detecta atajos de diseño).

## Licencia y créditos

Pendiente de definir la licencia del código propio. WHISKiT Physics pertenece al laboratorio SeNSE y mantiene su propia licencia; este repositorio solo incluye parches, no su código.
