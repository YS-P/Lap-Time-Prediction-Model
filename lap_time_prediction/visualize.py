# visualize.py
import matplotlib.pyplot as plt
import pandas as pd

team_colors = {
    'Red Bull Racing': '#3671C6',
    'Ferrari': '#DC0000',
    'Mercedes': '#00D2BE',
    'McLaren': '#FF8700',
    'Aston Martin': '#006F62',
    'Alpine': '#0090FF',
    'Williams': '#005AFF',
    'RB': '#1E41FF',
    'Kick Sauber': '#00E701',
    'Haas': '#B6BABD',
}

def add_team_info(data: pd.DataFrame, test: pd.DataFrame, pred_global, pred_driver=None):
    driver_to_team = data[['Driver', 'Team']].drop_duplicates().set_index('Driver')['Team'].to_dict()
    test_plot = test.copy()
    test_plot['PredGlobal'] = pred_global
    if pred_driver is not None:
        test_plot['PredDriver'] = pred_driver
    test_plot['Team'] = test_plot['Driver'].map(driver_to_team)
    test_plot['Color'] = test_plot['Team'].map(team_colors).fillna('#555555')
    return test_plot

def plot_driver_times(test_plot: pd.DataFrame, mae_per_drv=None, use_driver_specific=False):
    plt.style.use('ggplot')
    for drv in test_plot['Driver'].unique():
        drv_data = test_plot[test_plot['Driver'] == drv]
        color = drv_data['Color'].iloc[0]
        team_name = drv_data['Team'].iloc[0]
        mae_text = ""
        if mae_per_drv is not None and drv in mae_per_drv:
            mae_text = f"\nMAE: {mae_per_drv[drv]:.3f} sec"

        plt.figure(figsize=(12, 5))

        plt.plot(
            drv_data['LapTimeSec'].values,
            label=f'{drv} Actual',
            linewidth=2.6,
            color=color
        )

        if use_driver_specific and 'PredDriver' in drv_data:
            plt.plot(
                drv_data['PredDriver'].values,
                label=f'{drv} Driver-specific',
                linewidth=2.4,
                color=color,
                linestyle='--',
                alpha=0.8
            )

        plt.plot(
            drv_data['PredGlobal'].values,
            label=f'{drv} Global model',
            linewidth=2.2,
            color='black',
            linestyle=':',
            alpha=0.7
        )

        plt.title(f'Actual vs Predicted Lap Time — {drv} ({team_name}){mae_text}', fontsize=16)
        plt.xlabel('Lap Index', fontsize=13)
        plt.ylabel('Lap Time (sec)', fontsize=13)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
