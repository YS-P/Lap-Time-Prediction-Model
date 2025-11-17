# main.py
from load_data import load_session
from features import create_base_data, train_test_split_data, select_features, encode_train_test
from models import train_global_model, evaluate_global, mae_per_driver, train_driver_specific_models
from visualize import add_team_info, plot_driver_times

def main():
    merged = load_session()
    data = create_base_data(merged)
    train, test = train_test_split_data(data)

    selected_features, corr, top8 = select_features(train)
    X_train, y_train, X_test, y_test = encode_train_test(train, test, selected_features)

    # Global baseline
    model_global = train_global_model(X_train, y_train)
    pred_global, mae_global = evaluate_global(model_global, X_test, y_test)
    print(f"Global baseline MAE: {mae_global:.3f}")

    # Driver-specific models
    models_drv, pred_driver = train_driver_specific_models(train, X_train, y_train, test, X_test)

    # MAE per driver based on global model
    mae_drv = mae_per_driver(test, y_test, pred_global)
    print("MAE per driver (global model):")
    for drv, m in sorted(mae_drv.items(), key=lambda x: x[1]):
        print(f"{drv}: {m:.3f}")

    # Visualization
    test_plot = add_team_info(data, test, pred_global, pred_driver)
    plot_driver_times(test_plot, mae_per_drv=mae_drv, use_driver_specific=True)

if __name__ == "__main__":
    main()
