# test_complete.py
import tensorflow as tf
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt

print("✅ TESLA STOCK PREDICTION ENVIRONMENT TEST")
print("="*50)
print(f"✓ TensorFlow: {tf.__version__}")
print(f"✓ NumPy: {np.__version__}")
print(f"✓ Pandas: {pd.__version__}")
print(f"✓ Scikit-learn: {sklearn.__version__}")
print("="*50)

# Test TensorFlow functionality
try:
    x = tf.constant([1, 2, 3, 4])
    y = tf.square(x)
    print("✓ TensorFlow operations working!")
except Exception as e:
    print(f"✗ TensorFlow error: {e}")

# Test NumPy-TensorFlow integration
try:
    np_array = np.array([1, 2, 3])
    tf_tensor = tf.constant(np_array)
    result = tf.reduce_sum(tf_tensor)
    print("✓ NumPy-TensorFlow integration working!")
except Exception as e:
    print(f"✗ Integration error: {e}")

print("="*50)
print("🚀 Ready for Tesla Stock Prediction!")
