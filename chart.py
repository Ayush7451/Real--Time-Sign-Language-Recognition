import matplotlib.pyplot as plt
import numpy as np

# Sample accuracy results for sign language classes
labels = ["Hello", "No", "Thankyou", "Yes", "Love", "Victory", "Ok"]
accuracy = [94, 90, 92, 95, 93, 91, 96]

# Bar graph
plt.figure()
plt.bar(labels, accuracy)
plt.xlabel("Sign Classes")
plt.ylabel("Accuracy (%)")
plt.title("Class-wise Recognition Accuracy")
plt.show()

# Confusion matrix (sample)
confusion_matrix = np.array([
    [48,1,0,0,0,1,0],
    [2,45,1,0,0,0,2],
    [0,1,46,1,0,0,2],
    [0,0,1,47,1,0,1],
    [0,0,0,1,48,1,0],
    [1,0,0,0,2,46,1],
    [0,0,1,0,0,1,48]
])

plt.figure()
plt.imshow(confusion_matrix)
plt.xticks(range(len(labels)), labels, rotation=45)
plt.yticks(range(len(labels)), labels)
plt.title("Confusion Matrix")
plt.colorbar()
plt.show()