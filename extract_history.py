import json
import re

log_output = """
34/34 ==================== 56s 2s/step - accuracy: 0.3102 - loss: 1.8525 - val_accuracy: 0.2278 - val_loss: 3.6790 - learning_rate: 0.0010
34/34 ==================== 43s 1s/step - accuracy: 0.4852 - loss: 1.4425 - val_accuracy: 0.2444 - val_loss: 3.2395 - learning_rate: 0.0010
34/34 ==================== 40s 1s/step - accuracy: 0.5481 - loss: 1.2559 - val_accuracy: 0.2944 - val_loss: 2.0130 - learning_rate: 0.0010
34/34 ==================== 41s 1s/step - accuracy: 0.5694 - loss: 1.1227 - val_accuracy: 0.3556 - val_loss: 1.8032 - learning_rate: 0.0010
34/34 ==================== 47s 1s/step - accuracy: 0.6306 - loss: 1.0211 - val_accuracy: 0.5000 - val_loss: 1.4001 - learning_rate: 0.0010
34/34 ==================== 45s 1s/step - accuracy: 0.6769 - loss: 0.9400 - val_accuracy: 0.3722 - val_loss: 1.8564 - learning_rate: 0.0010
34/34 ==================== 42s 1s/step - accuracy: 0.6907 - loss: 0.8548 - val_accuracy: 0.4500 - val_loss: 1.5207 - learning_rate: 0.0010
34/34 ==================== 41s 1s/step - accuracy: 0.6981 - loss: 0.8517 - val_accuracy: 0.5278 - val_loss: 1.4381 - learning_rate: 0.0010
34/34 ==================== 32s 943ms/step - accuracy: 0.7491 - loss: 0.7367 - val_accuracy: 0.4833 - val_loss: 1.6323 - learning_rate: 0.0010
34/34 ==================== 40s 1s/step - accuracy: 0.7500 - loss: 0.6846 - val_accuracy: 0.4500 - val_loss: 1.7725 - learning_rate: 0.0010
34/34 ==================== 50s 1s/step - accuracy: 0.8222 - loss: 0.5661 - val_accuracy: 0.5444 - val_loss: 1.3444 - learning_rate: 5.0000e-04
34/34 ==================== 46s 1s/step - accuracy: 0.8315 - loss: 0.5136 - val_accuracy: 0.4889 - val_loss: 1.5290 - learning_rate: 5.0000e-04
34/34 ==================== 43s 1s/step - accuracy: 0.8500 - loss: 0.4879 - val_accuracy: 0.5389 - val_loss: 1.3850 - learning_rate: 5.0000e-04
34/34 ==================== 35s 1s/step - accuracy: 0.8231 - loss: 0.5076 - val_accuracy: 0.5889 - val_loss: 1.3448 - learning_rate: 5.0000e-04
34/34 ==================== 44s 1s/step - accuracy: 0.8657 - loss: 0.4229 - val_accuracy: 0.5611 - val_loss: 1.2728 - learning_rate: 5.0000e-04
34/34 ==================== 34s 997ms/step - accuracy: 0.8574 - loss: 0.4175 - val_accuracy: 0.5667 - val_loss: 1.2264 - learning_rate: 5.0000e-04
34/34 ==================== 33s 951ms/step - accuracy: 0.8750 - loss: 0.3863 - val_accuracy: 0.5389 - val_loss: 1.3264 - learning_rate: 5.0000e-04
34/34 ==================== 29s 870ms/step - accuracy: 0.8731 - loss: 0.3657 - val_accuracy: 0.5889 - val_loss: 1.1409 - learning_rate: 5.0000e-04
34/34 ==================== 38s 1s/step - accuracy: 0.8889 - loss: 0.3526 - val_accuracy: 0.6444 - val_loss: 1.1582 - learning_rate: 5.0000e-04
34/34 ==================== 42s 1s/step - accuracy: 0.9019 - loss: 0.3327 - val_accuracy: 0.5833 - val_loss: 1.2422 - learning_rate: 5.0000e-04
34/34 ==================== 40s 1s/step - accuracy: 0.8907 - loss: 0.3291 - val_accuracy: 0.6222 - val_loss: 1.2289 - learning_rate: 5.0000e-04
34/34 ==================== 42s 1s/step - accuracy: 0.9009 - loss: 0.3109 - val_accuracy: 0.5722 - val_loss: 1.2042 - learning_rate: 5.0000e-04
34/34 ==================== 44s 1s/step - accuracy: 0.8981 - loss: 0.3204 - val_accuracy: 0.5722 - val_loss: 1.2345 - learning_rate: 5.0000e-04
34/34 ==================== 41s 1s/step - accuracy: 0.9148 - loss: 0.2630 - val_accuracy: 0.6389 - val_loss: 1.1042 - learning_rate: 2.5000e-04
34/34 ==================== 39s 1s/step - accuracy: 0.9306 - loss: 0.2445 - val_accuracy: 0.6056 - val_loss: 1.1116 - learning_rate: 2.5000e-04
34/34 ==================== 45s 1s/step - accuracy: 0.9315 - loss: 0.2224 - val_accuracy: 0.6611 - val_loss: 1.0924 - learning_rate: 2.5000e-04
34/34 ==================== 42s 1s/step - accuracy: 0.9472 - loss: 0.2100 - val_accuracy: 0.6500 - val_loss: 1.1426 - learning_rate: 2.5000e-04
34/34 ==================== 77s 1s/step - accuracy: 0.9333 - loss: 0.2174 - val_accuracy: 0.6444 - val_loss: 1.0682 - learning_rate: 2.5000e-04
34/34 ==================== 38s 1s/step - accuracy: 0.9333 - loss: 0.2100 - val_accuracy: 0.6222 - val_loss: 1.1918 - learning_rate: 2.5000e-04
34/34 ==================== 37s 1s/step - accuracy: 0.9333 - loss: 0.2176 - val_accuracy: 0.6611 - val_loss: 1.0913 - learning_rate: 2.5000e-04
34/34 ==================== 38s 1s/step - accuracy: 0.9398 - loss: 0.1973 - val_accuracy: 0.6611 - val_loss: 1.0595 - learning_rate: 2.5000e-04
34/34 ==================== 42s 1s/step - accuracy: 0.9389 - loss: 0.2012 - val_accuracy: 0.6556 - val_loss: 1.0507 - learning_rate: 2.5000e-04
34/34 ==================== 43s 1s/step - accuracy: 0.9333 - loss: 0.1936 - val_accuracy: 0.6944 - val_loss: 1.0189 - learning_rate: 2.5000e-04
34/34 ==================== 80s 1s/step - accuracy: 0.9352 - loss: 0.2035 - val_accuracy: 0.6556 - val_loss: 1.1558 - learning_rate: 2.5000e-04
34/34 ==================== 42s 1s/step - accuracy: 0.9444 - loss: 0.1780 - val_accuracy: 0.6556 - val_loss: 1.1279 - learning_rate: 2.5000e-04
34/34 ==================== 37s 1s/step - accuracy: 0.9435 - loss: 0.1767 - val_accuracy: 0.7056 - val_loss: 1.0726 - learning_rate: 2.5000e-04
34/34 ==================== 36s 1s/step - accuracy: 0.9463 - loss: 0.1803 - val_accuracy: 0.6611 - val_loss: 1.1634 - learning_rate: 2.5000e-04
34/34 ==================== 41s 1s/step - accuracy: 0.9472 - loss: 0.1664 - val_accuracy: 0.7000 - val_loss: 1.0588 - learning_rate: 2.5000e-04
34/34 ==================== 31s 900ms/step - accuracy: 0.9620 - loss: 0.1383 - val_accuracy: 0.6944 - val_loss: 1.0482 - learning_rate: 1.2500e-04
34/34 ==================== 38s 1s/step - accuracy: 0.9556 - loss: 0.1604 - val_accuracy: 0.6944 - val_loss: 1.0378 - learning_rate: 1.2500e-04
34/34 ==================== 42s 1s/step - accuracy: 0.9519 - loss: 0.1464 - val_accuracy: 0.7056 - val_loss: 1.0739 - learning_rate: 1.2500e-04
34/34 ==================== 39s 1s/step - accuracy: 0.9380 - loss: 0.1669 - val_accuracy: 0.6722 - val_loss: 1.1450 - learning_rate: 1.2500e-04
34/34 ==================== 41s 1s/step - accuracy: 0.9593 - loss: 0.1454 - val_accuracy: 0.6944 - val_loss: 1.0899 - learning_rate: 1.2500e-04
34/34 ==================== 37s 1s/step - accuracy: 0.9704 - loss: 0.1197 - val_accuracy: 0.6833 - val_loss: 1.1211 - learning_rate: 6.2500e-05
34/34 ==================== 45s 1s/step - accuracy: 0.9630 - loss: 0.1384 - val_accuracy: 0.6722 - val_loss: 1.1191 - learning_rate: 6.2500e-05
34/34 ==================== 40s 1s/step - accuracy: 0.9676 - loss: 0.1180 - val_accuracy: 0.6889 - val_loss: 1.1143 - learning_rate: 6.2500e-05
34/34 ==================== 41s 1s/step - accuracy: 0.9667 - loss: 0.1345 - val_accuracy: 0.6944 - val_loss: 1.1007 - learning_rate: 6.2500e-05
34/34 ==================== 46s 1s/step - accuracy: 0.9676 - loss: 0.1219 - val_accuracy: 0.6944 - val_loss: 1.0876 - learning_rate: 6.2500e-05
34/34 ==================== 40s 1s/step - accuracy: 0.9620 - loss: 0.1310 - val_accuracy: 0.6944 - val_loss: 1.0667 - learning_rate: 3.1250e-05
34/34 ==================== 43s 1s/step - accuracy: 0.9704 - loss: 0.1148 - val_accuracy: 0.7056 - val_loss: 1.0503 - learning_rate: 3.1250e-05
"""

history = {
    'accuracy': [],
    'loss': [],
    'val_accuracy': [],
    'val_loss': [],
    'learning_rate': []
}

pattern = r'- accuracy: ([\d.]+) - loss: ([\d.]+) - val_accuracy: ([\d.]+) - val_loss: ([\d.]+) - learning_rate: ([\d.e-]+)'

for match in re.finditer(pattern, log_output):
    history['accuracy'].append(float(match.group(1)))
    history['loss'].append(float(match.group(2)))
    history['val_accuracy'].append(float(match.group(3)))
    history['val_loss'].append(float(match.group(4)))
    history['learning_rate'].append(float(match.group(5)))

with open('c:/Users/moham/OneDrive/Desktop/CodeAlpha/CodeAlpha_Emotion_Recognition/results/exp3_training_history.json', 'w') as f:
    json.dump(history, f, indent=4)
print("History saved to results/exp3_training_history.json")
