from deepface import build_model


class Net:

    def __init__(self, model_name):
        self.model = build_model(model_name)
