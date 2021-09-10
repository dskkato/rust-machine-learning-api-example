try:
    import tensorflow as tf
    if tf.__version__.split(".")[0] != "2":
        raise Exception("This script requires tensorflow >= 2")
except:
    print("// -----------------------\n[requirement] pip install tensorflow>=2\n// --------------------------")
    raise

# default input shape 224x224x3
model = tf.keras.applications.MobileNetV3Small(
    input_shape=(224, 224, 3), weights="imagenet"
)

# save the model
directory = "model"
model.save(directory, save_format="tf")
