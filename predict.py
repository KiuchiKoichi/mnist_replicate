import onnxruntime as ort
import numpy as np
from PIL import Image
from cog import BasePredictor, Path

class Predictor(BasePredictor):
    def setup(self):
        # ONNX モデルをロード
        self.session = ort.InferenceSession("model.onnx")
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def preprocess(self, image: Image) -> np.ndarray:
        # 画像の前処理（MNIST用の例）
        image = image.resize((28, 28)).convert('L')
        image = np.array(image).astype(np.float32)
        image = image.reshape(1, 1, 28, 28) / 255.0
        return image

    def postprocess(self, output: np.ndarray) -> int:
        # 推論結果の後処理
        return int(np.argmax(output))

    def predict(self, image: Path) -> int:
        # 画像をロード
        image = Image.open(image)

        # 前処理
        preprocessed_image = self.preprocess(image)

        # 推論
        output = self.session.run([self.output_name], {self.input_name: preprocessed_image})[0]

        # 後処理
        result = self.postprocess(output)
        return result
