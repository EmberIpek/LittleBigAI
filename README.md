# Technica 2025
# 11/15/2025
Self-driving vehicles sometimes suddenly brake for small objects, animals, and even shadows, putting lives of passengers in danger. Proposed solution is to build a CNN based on LeNet and train a ML model on CIFAR-10 dataset: {airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck}. When collision is imminent, use model to detect the type of object and decide whether to override braking based on the size of the object. CIFAR-10 used as a proxy dataset due to time constraints but can be modified for custom dataset to train for harmless vs dangerous objects.
Algorithm will consider {bird, cat, dog, frog} to be small objects and {airplane, automobile, deer, horse, ship, truck} to be large objects.

EmberNet structure:

<img width="975" height="499" alt="image" src="https://github.com/user-attachments/assets/c6cecd1f-6468-4cd8-ab00-d5012aede8be" />


## T-12 hours:
EmberNet implemented using PyTorch, CNN based on LeNet: cross-entropy loss function, SGD optimizer, lr = 0.01, batch size = 32. Training for 100 epochs with 10000 images:

<img width="473" height="371" alt="image" src="https://github.com/user-attachments/assets/63ee2b8f-934e-439c-802b-1f71c9b5dbab" /> <img width="496" height="363" alt="image" src="https://github.com/user-attachments/assets/b4504b9e-7bfb-4a18-8b88-3af0603b47c5" />

Test accuracy hovers around 50%. Added dropout layers to fully connected section to fix overfitting, changed linear flattening layers to ReLU and added dropout layers. Training for 50 epochs with 50000 images:

<img width="481" height="366" alt="image" src="https://github.com/user-attachments/assets/eb39f0ed-0777-495e-a426-40319fa5571c" /> <img width="473" height="365" alt="image" src="https://github.com/user-attachments/assets/50b64255-c9ae-4f75-8f87-381d3041dfc0" />
   
Test accuracy ~60%, will use current model for hackathon purposes.

## T-9 hours:
EmberNet exported to ONNX file, loaded on RPi. RPi camera capture using OpenCV, Haar Cascade face detection implemented as test:

<img width="975" height="440" alt="image" src="https://github.com/user-attachments/assets/370decca-2fba-4327-a0e7-7a3104926eb3" />

## T-6 hours:
Convert PyTorch classification model EmberNet to ONNX format, run converted PyTorch model with OpenCV. OpenCV having issue with ONNX model, use ONNX runtime:

```python
import onnxruntime as ort

ort_session = ort.InferenceSession("EmberNet.onnx")
```

## T-3 hours:
Scanning each frame for big objects results in model hallucinating trucks everywhere. Using higher confidence threshold results in no objects being detected, very low frame rate when attempting to scan with multiple different sized blocks.

## T-1 hour:
I’m a truck. Everything is a truck:

<img width="975" height="397" alt="image" src="https://github.com/user-attachments/assets/6a77edc1-e34c-4912-a5dc-8670deccef54" />

<img width="975" height="392" alt="image" src="https://github.com/user-attachments/assets/6aa9c04a-a599-48ea-9013-e711533fad26" />

## T-5 minutes:

HORSE DETECTED:

<img width="975" height="778" alt="image" src="https://github.com/user-attachments/assets/605e988c-3d04-4b0b-97c7-fac4c94bb7d2" />

