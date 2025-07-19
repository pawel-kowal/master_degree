import matplotlib.pyplot as plt
import seaborn as sns
import shap

def plot_predictions(y_true, y_pred, title='Prognoza vs Rzeczywistość'):
    plt.figure(figsize=(12, 6))
    plt.plot(y_true.index, y_true, label='Rzeczywistość', color='blue')
    plt.plot(y_true.index, y_pred, label='Prognoza', color='red', linestyle='--')
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

def plot_multiple_preds(y_true, predictions: dict, title='Porównanie prognoz vs Rzeczywistość'):
    """
    Wyświetla wykres porównawczy rzeczywistych wartości z wieloma prognozami.

    Args:
        y_true (pd.Series): Rzeczywiste wartości (z indeksem czasowym).
        predictions (dict[str, pd.Series]): Słownik, gdzie klucz to nazwa modelu, a wartość to seria prognoz.
        title (str): Tytuł wykresu.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(y_true.index, y_true, label='Rzeczywistość', color='black', linewidth=2)
    for name, y_pred in predictions.items():
        plt.plot(y_pred.index, y_pred, label=name)
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

def plot_feature_importance(model, feature_names):
    importance = model.feature_importances_
    sorted_idx = importance.argsort()
    plt.figure(figsize=(12, 8))
    plt.barh(np.array(feature_names)[sorted_idx], importance[sorted_idx])
    plt.title('Feature Importance')
    plt.xlabel('Importance')
    plt.tight_layout()
    plt.show()

def plot_shap_summary(model, X_scaled, feature_names):
    explainer = shap.Explainer(model)
    shap_values = explainer(X_scaled)
    shap.summary_plot(shap_values, X_scaled, feature_names=feature_names)
