import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Evita erro de interface gráfica GTK/GNOME no Linux
import matplotlib.pyplot as plt

# Configuração visual limpa e profissional para publicação acadêmica
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['grid.color'] = '#eeeeee'
plt.rcParams['grid.linestyle'] = '--'

# Função auxiliar flexível para carregar cada dataset (suporta subpastas ou pasta raiz)
def load_csv(primary_name, fallback_paths):
    for path in [primary_name] + fallback_paths:
        if os.path.exists(path):
            print(f"📁 Carregando dataset: {path}")
            return pd.read_csv(path)
    raise FileNotFoundError(f"Não foi possível encontrar o arquivo '{primary_name}'. Verifique o caminho.")

# 1. Carregamento dos dados
# Carregamento direto conforme as subpastas da sua máquina
df_llama = pd.read_csv('summary_data/benchmark_summary_faults_llama.csv')
df_qwen = pd.read_csv('summary_data/benchmark_summary_faults_qwen.csv')
df_gemini = pd.read_csv('summary_data/benchmark_summary_faults_gemini.csv')

# Padronização de rótulos dos modelos
df_llama['model_label'] = 'Llama 3.1 8B (Local)'
df_qwen['model_label'] = 'Qwen 2.5 Coder 7B (Local)'
df_gemini['model_label'] = 'Gemini 3.1 Flash Lite (Nuvem)'

# Consolidação em dataset único
df_all = pd.concat([df_llama, df_qwen, df_gemini], ignore_index=True)

# Mapeamento de cargas de trabalho e cenários de falha
workload_map = {
    'Carga_A': 'Carga A',
    'Carga_B': 'Carga B',
    'Carga_C': 'Carga C'
}
df_all['workload_label'] = df_all['workload'].map(workload_map)

scenario_map = {
    'baseline': 'Baseline',
    'crash_bug': 'Crash Bug',
    'stuck_bug': 'Stuck Bug',
    'logical_bug': 'Logical Bug',
    'visual_bug': 'Visual Bug'
}
df_all['scenario_label'] = df_all['fault_scenario'].map(scenario_map)

# Paleta de cores acessível (Okabe-Ito)
colors = {
    'Llama 3.1 8B (Local)': '#0072b2',       # Azul
    'Qwen 2.5 Coder 7B (Local)': '#009e73',  # Verde
    'Gemini 3.1 Flash Lite (Nuvem)': '#e69f00' # Laranja
}

model_order = ['Llama 3.1 8B (Local)', 'Qwen 2.5 Coder 7B (Local)', 'Gemini 3.1 Flash Lite (Nuvem)']
workload_order = ['Carga A', 'Carga B', 'Carga C']
scenario_order = ['Baseline', 'Crash Bug', 'Stuck Bug', 'Logical Bug', 'Visual Bug']

os.makedirs('figures', exist_ok=True)

# =============================================================================
# GRÁFICO 1: LATÊNCIA MÉDIA POR CARGA DE TRABALHO
# =============================================================================
try:
    fig1, ax1 = plt.subplots(figsize=(10, 6), dpi=300)
    df_grouped = df_all.groupby(['workload_label', 'model_label'])['latency_seconds'].mean().reset_index()

    x1 = np.arange(len(workload_order))
    width1 = 0.25

    for i, model in enumerate(model_order):
        sub = df_grouped[df_grouped['model_label'] == model]
        means = [sub[sub['workload_label'] == w]['latency_seconds'].values[0] if not sub[sub['workload_label'] == w].empty else 0 for w in workload_order]
        offset = (i - 1) * width1
        rects = ax1.bar(x1 + offset, means, width1, label=model, color=colors[model])
        
        for rect in rects:
            h = rect.get_height()
            if h > 0:
                ax1.annotate(f'{h:.1f}s',
                            xy=(rect.get_x() + rect.get_width() / 2, h),
                            xytext=(0, 4), textcoords="offset points",
                            ha='center', va='bottom', fontsize=9, fontweight='bold', color=colors[model])

    ax1.set_xticks(x1)
    ax1.set_xticklabels(workload_order, fontsize=10.5)
    ax1.set_ylabel('Latência Média (segundos)', fontsize=11, fontweight='bold')
    ax1.set_ylim(0, max(df_grouped['latency_seconds']) * 1.18)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    ax1.set_title("Comparativo de Latência Média por Carga de Trabalho\n"
                  "Modelos Locais vs. Modelo em Nuvem (N = 45 por modelo)",
                  fontsize=12, fontweight='bold', pad=45, loc='center')

    ax1.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=False, fontsize=9.5)

    plt.tight_layout()
    path1 = 'figures/benchmark_latency_comparison.png'
    plt.savefig(path1, dpi=300, bbox_inches='tight')
    plt.close(fig1)
    print("Gráfico 1 gerado: benchmark_latency_comparison.png")
except Exception as e:
    print(f"❌ Erro no Gráfico 1: {e}")

# =============================================================================
# GRÁFICO 2: TAXAS DE ACURÁCIA E CONFORMIDADE
# =============================================================================
try:
    fig2, ax2 = plt.subplots(figsize=(10, 6), dpi=300)

    acc_metrics = {
        'Formato Válido': 'valid_format',
        'Parseável/Compilável': 'compilable_parseable',
        'Detecção de Defeito': 'defect_detected'
    }

    acc_data = []
    for model in model_order:
        sub = df_all[df_all['model_label'] == model]
        for label, col in acc_metrics.items():
            if col in sub.columns:
                val = pd.to_numeric(sub[col], errors='coerce').fillna(0).mean() * 100
            else:
                val = 0
            acc_data.append({'model_label': model, 'metric': label, 'pct': val})

    df_acc = pd.DataFrame(acc_data)
    x2 = np.arange(len(acc_metrics))
    width2 = 0.25

    for i, model in enumerate(model_order):
        sub = df_acc[df_acc['model_label'] == model]
        vals = sub['pct'].values
        offset = (i - 1) * width2
        rects = ax2.bar(x2 + offset, vals, width2, label=model, color=colors[model])
        
        for rect in rects:
            h = rect.get_height()
            ax2.annotate(f'{h:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 4), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold', color=colors[model])

    ax2.set_xticks(x2)
    ax2.set_xticklabels(list(acc_metrics.keys()), fontsize=10.5)
    ax2.set_ylabel('Taxa de Sucesso (%)', fontsize=11, fontweight='bold')
    ax2.set_ylim(0, 125)
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    ax2.set_title("Taxa de Sucesso, Conformidade Sintática e Detecção de Defeitos\n"
                  "Avaliação de qualidade sob Injeção de Falha (N = 45 por modelo)",
                  fontsize=12, fontweight='bold', pad=45, loc='center')

    ax2.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=False, fontsize=9.5)

    plt.tight_layout()
    path2 = 'figures/benchmark_accuracy_comparison.png'
    plt.savefig(path2, dpi=300, bbox_inches='tight')
    plt.close(fig2)
    print("Gráfico 2 gerado: benchmark_accuracy_comparison.png")
except Exception as e:
    print(f"❌ Erro no Gráfico 2: {e}")

# =============================================================================
# GRÁFICO 3: LATÊNCIA MÉDIA POR CENÁRIO DE INJEÇÃO DE FALHA
# =============================================================================
try:
    fig3, ax3 = plt.subplots(figsize=(11, 6), dpi=300)
    df_scen_grouped = df_all.groupby(['scenario_label', 'model_label'])['latency_seconds'].mean().reset_index()

    x3 = np.arange(len(scenario_order))
    width3 = 0.25

    for i, model in enumerate(model_order):
        sub = df_scen_grouped[df_scen_grouped['model_label'] == model]
        means = [sub[sub['scenario_label'] == s]['latency_seconds'].values[0] if not sub[sub['scenario_label'] == s].empty else 0 for s in scenario_order]
        offset = (i - 1) * width3
        rects = ax3.bar(x3 + offset, means, width3, label=model, color=colors[model])
        
        for rect in rects:
            h = rect.get_height()
            if h > 0:
                ax3.annotate(f'{h:.1f}s',
                            xy=(rect.get_x() + rect.get_width() / 2, h),
                            xytext=(0, 4), textcoords="offset points",
                            ha='center', va='bottom', fontsize=8.5, fontweight='bold', color=colors[model])

    ax3.set_xticks(x3)
    ax3.set_xticklabels(scenario_order, fontsize=10.5)
    ax3.set_ylabel('Latência Média (segundos)', fontsize=11, fontweight='bold')
    ax3.set_ylim(0, max(df_scen_grouped['latency_seconds']) * 1.18)
    ax3.grid(axis='y', linestyle='--', alpha=0.7)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)

    ax3.set_title("Sensibilidade de Latência por Cenário de Injeção de Falhas\n"
                  "Tempo de resposta médio em segundos por tipo de erro injetado",
                  fontsize=12, fontweight='bold', pad=45, loc='center')

    ax3.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=False, fontsize=9.5)

    plt.tight_layout()
    path3 = 'figures/benchmark_scenario_latency.png'
    plt.savefig(path3, dpi=300, bbox_inches='tight')
    plt.close(fig3)
    print("Gráfico 3 gerado: benchmark_scenario_latency.png")
except Exception as e:
    print(f"❌ Erro no Gráfico 3: {e}")

print("Todos os 3 gráficos foram processados com sucesso!")
