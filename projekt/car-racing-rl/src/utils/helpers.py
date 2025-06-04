def preprocess_observation(observation):
    # Przykładowa funkcja do przetwarzania obserwacji
    # Można dodać kod do normalizacji, zmiany rozmiaru itp.
    return observation

def save_model(model, filename):
    # Funkcja do zapisywania modelu
    import joblib
    joblib.dump(model, filename)