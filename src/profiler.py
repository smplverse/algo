from deepface import DeepFace
from model_profiler import model_profiler


def run_profiler():
    model_names = [
        'VGG-Face',
        'OpenFace',
        'Facenet',
        'Facenet512',
        'DeepFace',
        'DeepID',
        'ArcFace',
    ]
    for model_name in model_names:
        model = DeepFace.build_model(model_name)
        profile = model_profiler(model, 1)
        print("model_name:", model_name)
        print(profile, "\n")


if __name__ == "__main__":
    run_profiler()
