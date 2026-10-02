import joblib
from pathlib import Path

def save_model_artifacts(model, preprocessor, feature_names):

    # project root
    root_dir = Path(__file__).resolve().parent.parent

    # app folder
    app_dir = root_dir / 'app'

    # create app folder if missing
    app_dir.mkdir(exist_ok=True)

    # save artifacts
    joblib.dump(model, app_dir / 'model.pkl')
    joblib.dump(preprocessor, app_dir / 'preprocessor.pkl')
    joblib.dump(feature_names, app_dir / 'feature_names.pkl')

    print("Model artifacts saved successfully!")