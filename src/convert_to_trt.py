from tensorflow.python.compiler.tensorrt import trt_convert as trt

input_saved_model_dir = "/home/piotrostr/.deepface/weights/VGGFace2_DeepFace_weights_val-0.9034.h5"
input_saved_model_dir
precision_mode = trt.TrtPrecisionMode.FP32

conversion_params = trt.TrtConversionParams(precision_mode=precision_mode)

converter = trt.TrtGraphConverterV2(
    input_saved_model_dir=input_saved_model_dir,
    conversion_params=conversion_params,
)

# Converter method used to partition and optimize TensorRT compatible segments
converter.convert()

# Optionally, build TensorRT engines before deployment to save time at runtime
# Note that this is GPU specific, and as a rule of thumb, we recommend building at runtime
# converter.build(input_fn=my_input_fn)

# Save the model to the disk
converter.save("VGGFace.trt")
