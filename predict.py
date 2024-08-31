import onnxruntime as ort
import numpy as np

def preprocess(image):
    # 画像の前処理（MNIST用の例）
    image = image.resize((28, 28)).convert('L')
    image = np.array(image).astype(np.float32)
    image = image.reshape(1, 1, 28, 28) / 255.0
    return image

def postprocess(output):
    # 推論結果の後処理
    return np.argmax(output)

def predict(image):
    session = ort.InferenceSession("model.onnx")
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    preprocessed_image = preprocess(image)
    output = session.run([output_name], {input_name: preprocessed_image})[0]

    result = postprocess(output)
    return result
