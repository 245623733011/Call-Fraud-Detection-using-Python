import pickle
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Sample dummy data
X = np.array([
    [30, 0, 0],
    [180, 1, 1],
    [20, 0, 0],
    [300, 1, 1]
])
y = np.array([0, 1, 0, 1])  # 0 = Normal, 1 = Fraud

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Save model
with open('fraud_detection_model.pkl', 'wb') as f:
    pickle.dump(model, f)
