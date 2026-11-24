## Como Executar o Projeto

Esta seção apresenta o passo a passo para configurar o ambiente, executar a matriz experimental do benchmarking e gerar as figuras comparativas utilizadas no TCC.

---

### 1. Pré-Requisitos e Instalação

#### 1.1 Requisitos do Sistema
* **Python 3.10+**
* **Ollama** (para execução dos modelos locais no Linux/WSL)

#### 1.2 Baixar os Modelos Locais no Ollama
Certifique-se de que o serviço do Ollama está rodando e faça o download dos modelos utilizados na pesquisa:
```bash
ollama pull llama3.1:8b
ollama pull qwen2.5-coder:7b
```

#### 1.3 Instalar as Bibliotecas Python
Instale as dependências necessárias para automação e geração de estatísticas/gráficos:

```bash
pip install pandas matplotlib numpy requests
```

### 2.  Execução do Benchmarking (```benchmark-naev-qa.py```)
O script principal gerencia as 135 execuções empíricas (3 modelos × 3 cargas de trabalho × 5 cenários de falha × 3 repetições).

#### 2.1 Executar com Llama 3.1 8B (Local)

```bash
python3 benchmark-naev-qa.py --real --model llama --runs 3
```

##### 2.2 Executar com Qwen 2.5 Coder 7B (Local)

```bash
python3 benchmark-naev-qa.py --real --model qwen --runs 3
```

#### 2.3 Executar com Gemini 3.1 Flash Lite (Nuvem)
Para executar a avaliação do modelo comercial em nuvem, exporte a sua chave de API do Google AI Studio e passe explicitamente o modelo `gemini-3.1-flash-lite` via parâmetro:

```bash
export GEMINI_API_KEY="SUA_CHAVE_DE_API_AQUI"
python3 benchmark-naev-qa.py --real --model gemini --runs 3 --gemini-model gemini-3.1-flash-lite

```

**Saídas geradas:** Ao final das execuções, o script salva os dados brutos nos arquivos ```benchmark_results_faults.json``` e os resumos estatísticos em ```benchmark_summary_faults.csv```.

### 3. Geração dos gráficos (```generate_all_charts.py```)
Para processar os dados dos arquivos CSV e gerar as 3 figuras comparativas em altíssima resolução (300 DPI), execute:

```bash
MPLBACKEND=Agg python3 generate_all_charts.py
```

#### Imagens produzidas na pasta do projeto:
```benchmark_latency_comparison.png```: Latência média comparativa por Carga de Trabalho (A, B e C).
```benchmark_accuracy_comparison.png```: Taxas de formato válido, compilabilidade e acurácia de detecção de defeitos.
```benchmark_scenario_latency.png```:Sensibilidade de latência separada pelos cinco cenários de injeção de falhas.

