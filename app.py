from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# ── Load all artefacts ──────────────────────────────────────────────────────
BASE = os.path.join(os.path.dirname(__file__), 'models')

model            = joblib.load(os.path.join(BASE, 'depression_model.pkl'))
label_encoders   = joblib.load(os.path.join(BASE, 'label_encoders.pkl'))
pca_transformer  = joblib.load(os.path.join(BASE, 'pca_transformer.pkl'))
scaler           = joblib.load(os.path.join(BASE, 'scaler.pkl'))
selected_features = joblib.load(os.path.join(BASE, 'selected_features.pkl'))

# Choices exposed to the form
CITIES      = list(label_encoders['City'].classes_)
PROFESSIONS = list(label_encoders['Profession'].classes_)
DEGREES     = list(label_encoders['Degree'].classes_)

SLEEP_OPTIONS = [
    "'Less than 5 hours'",
    "'5-6 hours'",
    "'7-8 hours'",
    "'More than 8 hours'"
]
DIETARY_OPTIONS = ['Healthy', 'Moderate', 'Unhealthy']

# ── Routes ──────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template(
        'index.html',
        cities=CITIES,
        professions=PROFESSIONS,
        degrees=DEGREES,
        sleep_options=SLEEP_OPTIONS,
        dietary_options=DIETARY_OPTIONS
    )


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # ── 1. Raw input ────────────────────────────────────────────────────
        gender              = data['gender']
        age                 = float(data['age'])
        city                = data['city']
        profession          = data['profession']
        academic_pressure   = float(data['academic_pressure'])
        work_pressure       = float(data['work_pressure'])
        cgpa                = float(data['cgpa'])
        study_satisfaction  = float(data['study_satisfaction'])
        job_satisfaction    = float(data['job_satisfaction'])
        sleep_duration      = data['sleep_duration']
        dietary_habits      = data['dietary_habits']
        degree              = data['degree']
        suicidal_thoughts   = data['suicidal_thoughts']
        work_study_hours    = float(data['work_study_hours'])
        financial_stress    = float(data['financial_stress'])
        family_history      = data['family_history']

        # ── 2. Label-encode categorical cols ────────────────────────────────
        gender_enc     = label_encoders['Gender'].transform([gender])[0]
        city_enc       = label_encoders['City'].transform([city])[0]
        profession_enc = label_encoders['Profession'].transform([profession])[0]
        degree_enc     = label_encoders['Degree'].transform([degree])[0]
        suicidal_enc   = label_encoders['Have you ever had suicidal thoughts ?'].transform([suicidal_thoughts])[0]
        family_enc     = label_encoders['Family History of Mental Illness'].transform([family_history])[0]

        # ── 3. One-hot for Sleep Duration & Dietary Habits ──────────────────
        sleep_less5   = 1 if sleep_duration == "'Less than 5 hours'" else 0
        dietary_mod   = 1 if dietary_habits == 'Moderate' else 0
        dietary_unh   = 1 if dietary_habits == 'Unhealthy' else 0

        # ── 4. Engineered feature ───────────────────────────────────────────
        total_pressure = academic_pressure + work_pressure

        # ── 5. Additional engineered features ───────────────────────────────
        satisfaction_ratio = (study_satisfaction + job_satisfaction) / (academic_pressure + work_pressure + 1)

        sleep_7_8      = 1 if sleep_duration == "'7-8 hours'"        else 0
        sleep_more8    = 1 if sleep_duration == "'More than 8 hours'" else 0
        sleep_others   = 1 if sleep_duration not in (
                                "'Less than 5 hours'", "'5-6 hours'",
                                "'7-8 hours'", "'More than 8 hours'") else 0
        dietary_others = 0  # not present in our dropdown

        # ── 6. Build full 23-feature DataFrame (must match scaler columns) ──
        SCALER_COLS = [
            'Gender', 'Age', 'City', 'Profession',
            'Academic Pressure', 'Work Pressure', 'CGPA',
            'Study Satisfaction', 'Job Satisfaction', 'Degree',
            'Have you ever had suicidal thoughts ?', 'Work/Study Hours',
            'Financial Stress', 'Family History of Mental Illness',
            'TotalPressure', 'SatisfactionRatio',
            "'Sleep Duration_'7-8 hours'", "'Sleep Duration_'Less than 5 hours'",
            "'Sleep Duration_'More than 8 hours'", 'Sleep Duration_Others',
            'Dietary Habits_Moderate', 'Dietary Habits_Others',
            'Dietary Habits_Unhealthy',
        ]
        full_row = {
            'Gender':                                  gender_enc,
            'Age':                                     age,
            'City':                                    city_enc,
            'Profession':                              profession_enc,
            'Academic Pressure':                       academic_pressure,
            'Work Pressure':                           work_pressure,
            'CGPA':                                    cgpa,
            'Study Satisfaction':                      study_satisfaction,
            'Job Satisfaction':                        job_satisfaction,
            'Degree':                                  degree_enc,
            'Have you ever had suicidal thoughts ?':   suicidal_enc,
            'Work/Study Hours':                        work_study_hours,
            'Financial Stress':                        financial_stress,
            'Family History of Mental Illness':        family_enc,
            'TotalPressure':                           total_pressure,
            'SatisfactionRatio':                       satisfaction_ratio,
            "Sleep Duration_'7-8 hours'":              sleep_7_8,
            "Sleep Duration_'Less than 5 hours'":      sleep_less5,
            "Sleep Duration_'More than 8 hours'":      sleep_more8,
            'Sleep Duration_Others':                   sleep_others,
            'Dietary Habits_Moderate':                 dietary_mod,
            'Dietary Habits_Others':                   dietary_others,
            'Dietary Habits_Unhealthy':                dietary_unh,
        }

        # ── 7. Scale using original scaler feature ordering ─────────────────
        X_df = pd.DataFrame([full_row])[list(scaler.feature_names_in_)]
        X_scaled = scaler.transform(X_df)

        # ── 7b. Select features for PCA ─────────────────────────────────────
        selected_idx = [list(scaler.feature_names_in_).index(f) for f in selected_features]
        X_selected = X_scaled[:, selected_idx]

        # ── 8. PCA ──────────────────────────────────────────────────────────
        X_pca = pca_transformer.transform(X_selected)

        # ── 9. Predict ──────────────────────────────────────────────────────
        prediction   = model.predict(X_pca)[0]
        probability  = model.predict_proba(X_pca)[0]

        result = {
            'prediction': int(prediction),
            'label': 'Depressed' if prediction == 1 else 'Not Depressed',
            'confidence': float(max(probability)) * 100,
            'prob_depressed': float(probability[1]) * 100,
            'prob_not_depressed': float(probability[0]) * 100,
        }
        return jsonify({'success': True, 'result': result})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True, port=5000)
