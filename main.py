import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8')

sns.set_palette("husl")


def load_data(path='cardio_v2.csv'):
    df = pd.read_csv(path, sep=';')
    return df


def check_missing_and_invalid(df):
    report = {}
    report['missing'] = df.isna().sum().to_dict()

    report['duplicate_rows'] = int(df.duplicated().sum())

    # Valores incorretos
    report['height_invalid'] = int(df[(df['height'] < 120) | (df['height'] > 230)].shape[0])
    report['weight_invalid'] = int(df[(df['weight'] < 30) | (df['weight'] > 250)].shape[0])
    report['ap_lo_high'] = int(df[df['ap_lo'] > df['ap_hi']].shape[0])
    report['age_outlier'] = int(df[(df['age'] <= 0) | (df['age'] > 120*365)].shape[0])

    return report


def clean_data(df):
    df = df.copy()

    df = df.dropna()

    df = df[~df.duplicated()]
    df = df[(df['height'] >= 120) & (df['height'] <= 230)]
    df = df[(df['weight'] >= 30) & (df['weight'] <= 250)]
    df = df[(df['ap_lo'] > 0) & (df['ap_hi'] > 0)]
    df = df[df['ap_lo'] <= df['ap_hi']]
    df = df[(df['age'] > 0) & (df['age'] <= 120*365)]
    df = df[df['gender'].isin([1, 2])]
    df = df[df['cholesterol'].isin([1, 2, 3])]
    df = df[df['gluc'].isin([1, 2, 3])]
    df = df[df['smoke'].isin([0, 1])]
    df = df[df['alco'].isin([0, 1])]
    df = df[df['active'].isin([0, 1])]
    df = df[df['cardio'].isin([0, 1])]

    df['age_years'] = (df['age'] / 365.25).round(1)

    df['bmi'] = (df['weight'] / (df['height'] / 100) ** 2).round(2)

    return df


def compute_summary(df):
    s = {}
    s['total'] = len(df)
    s['by_gender'] = df['gender'].value_counts().to_dict()

    s['height'] = {
        'min': float(df['height'].min()),
        'max': float(df['height'].max()),
        'mean': float(df['height'].mean()),
    }
    s['weight'] = {
        'min': float(df['weight'].min()),
        'max': float(df['weight'].max()),
        'mean': float(df['weight'].mean()),
    }
    s['age_days'] = {
        'min': int(df['age'].min()),
        'max': int(df['age'].max()),
        'mean': float(df['age'].mean()),
    }
    s['age_years'] = {
        'min': float(df['age_years'].min()),
        'max': float(df['age_years'].max()),
        'mean': float(df['age_years'].mean()),
    }
    s['blood_pressure'] = {
        'ap_lo_mean': float(df['ap_lo'].mean()),
        'ap_hi_mean': float(df['ap_hi'].mean()),
    }
    s['cardio_count'] = int(df['cardio'].sum())

    counts = df['cardio'].value_counts()
    s['cardio_balance'] = counts.to_dict()
    s['cardio_ratio'] = {
        '0': float(counts.get(0, 0)/len(df)),
        '1': float(counts.get(1, 0)/len(df)),
    }

    return s


def plot_cardio_balance(df):
    ax = sns.countplot(data=df, x='cardio')
    ax.set_title('Distribuição de Cardiovascular (0=não, 1=sim)')
    plt.tight_layout()
    plt.savefig('cardio_balance.png')
    plt.clf()


def plot_relation(df, feature, title=None, kind='count'):
    if kind == 'count':
        ax = sns.countplot(data=df, x=feature, hue='cardio')
    else:
        ax = sns.boxplot(data=df, x='cardio', y=feature)
    ax.set_title(title or f'{feature} vs cardio')
    plt.tight_layout()
    plt.savefig(f'{feature}_vs_cardio.png')
    plt.clf()


def print_make_report(summary, missing_report):
    print('\nDados ausentes - incorretos:')
    for k, v in missing_report.items():
        print(f'{k}: {v}')

    print('\nEstatísticas resumidas:')
    print(f"Total de registros: {summary['total']}")
    print(f"Contagem por gênero: {summary['by_gender']}")
    print(f"Altura (mín/máx/média): {summary['height']}")
    print(f"Peso (mín/máx/média): {summary['weight']}")
    print(f"Idade em dias (mín/máx/média): {summary['age_days']}")
    print(f"Idade em anos (mín/máx/média): {summary['age_years']}")
    print(f"Pressão diastólica média (ap_lo): {summary['blood_pressure']['ap_lo_mean']:.2f}")
    print(f"Pressão sistólica média (ap_hi): {summary['blood_pressure']['ap_hi_mean']:.2f}")
    print(f"Casos positivos de cardio: {summary['cardio_count']} ({summary['cardio_ratio']})")
    print('\nClassificável?')
    print('Sim, alvo binário cardio. Balanceamento: ' + str(summary['cardio_balance']))


if __name__ == '__main__':
    df = load_data('cardio_v2.csv')

    missing_report = check_missing_and_invalid(df)
    df_clean = clean_data(df)
    summary = compute_summary(df_clean)

    print_make_report(summary, missing_report)

    plot_cardio_balance(df_clean)
    plot_relation(df_clean, 'cholesterol', 'Colesterol vs Doença Cardiovascular')
    plot_relation(df_clean, 'smoke', 'Fumantes vs Doença Cardiovascular')
    plot_relation(df_clean, 'active', 'Atividade física vs Doença Cardiovascular')
    plot_relation(df_clean, 'gluc', 'Glicose vs Doença Cardiovascular')
    plot_relation(df_clean, 'bmi', 'IMC vs Doença Cardiovascular', kind='box')

    print('\nGráficos salvos: cardio_balance.png, cholesterol_vs_cardio.png, smoke_vs_cardio.png, active_vs_cardio.png, gluc_vs_cardio.png, bmi_vs_cardio.png')
