def predict_gen(meta1):
    import pickle
    import os
    import pandas as pd
    from django.conf import settings

    path = os.path.join(settings.MODELS, 'best_music_genre_model.pkl')

    with open(path, 'rb') as pickled:
        model_package = pickle.load(pickled)

    model = model_package["model"]
    model_name = model_package["model_name"]
    feature_columns = model_package["feature_columns"]

    # meta1 should be a dictionary returned by getmetadata()
    # Build DataFrame in exact training feature order
    sample_df = pd.DataFrame([[meta1[col] for col in feature_columns]], columns=feature_columns)

    if model_name == "Tuned SVM + SelectKBest":
        selector = model_package["selector"]
        scaler = model_package["scaler"]

        sample_selected = selector.transform(sample_df)
        sample_scaled = scaler.transform(sample_selected)
        pred = model.predict(sample_scaled)[0]

    elif model_name in ["Tuned SVM", "Tuned KNN"]:
        scaler = model_package["scaler"]
        sample_scaled = scaler.transform(sample_df)
        pred = model.predict(sample_scaled)[0]

    else:
        pred = model.predict(sample_df)[0]

    return pred