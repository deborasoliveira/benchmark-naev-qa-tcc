#!/usr/bin/env python3
"""
Framework de Benchmarking de QA para o Jogo Naev (v0.13.5) - Versão 12
----------------------------------------------------------------------
Automatiza a execução da matriz de Benchmarking de LLMs com Fault Seeding.

Destaques da v12:
  - Suporte ao endpoint /v1 e /v1beta da Generative Language API do Google.
  - Seleção inteligente de modelo Gemini (`gemini-1.5-flash`, `gemini-1.5-pro`, `gemini-2.0-flash`).
  - Tratamento automático de retentativas para erros de alta demanda (HTTP 503/429).
  - Timeout estendido de 300s para o Ollama local.
  - Acúmulo seguro de resultados nos arquivos CSV e JSON existentes.

Uso:
  python3 benchmark-naev-qa-v12.py --real --model gemini --runs 3 --gemini-model gemini-1.5-flash
"""

import os
import sys
import json
import time
import re
import argparse
import urllib.request
import urllib.error
from typing import Dict, Any, List, Optional
from datetime import datetime

# ==============================================================================
# CONFIGURAÇÃO DOS MODELOS E SCENARIOS PADRONIZADOS
# ==============================================================================

DEFAULT_GEMINI_MODEL = "gemini-1.5-flash"

ALL_MODELS_CONFIG = {
    "Llama 3.1 8B": {
        "provider": "ollama",
        "model_id": "llama3.1:8b",
        "key_name": "llama"
    },
    "Qwen 2.5 Coder 7B": {
        "provider": "ollama",
        "model_id": "qwen2.5-coder:7b",
        "key_name": "qwen"
    },
    "Gemini Flash": {
        "provider": "google",
        "model_id": DEFAULT_GEMINI_MODEL,
        "env_key": "GEMINI_API_KEY",
        "key_name": "gemini"
    }
}

FAULT_SCENARIOS = {
    "baseline": {
        "title": "Baseline (Sem Injeção de Falhas)",
        "description": "Script original da missão nelly01 sem defeitos injetados.",
        "file_name": "nelly01_baseline.lua"
    },
    "crash_bug": {
        "title": "Crash Bug (Exceção por Nil Value)",
        "description": "Injeção de acesso a piloto_nulo na função land() gerando erro de runtime.",
        "file_name": "nelly01_crash.lua"
    },
    "stuck_bug": {
        "title": "Stuck Bug (Trava de Progressão)",
        "description": "Validação com requisito de item inexistente (Item_Fantasma_999).",
        "file_name": "nelly01_stuck.lua"
    },
    "logical_bug": {
        "title": "Logical Bug (Erro de Recompensa)",
        "description": "Dedução de créditos player.pay(-reward) no lugar da recompensa.",
        "file_name": "nelly01_logical.lua"
    },
    "visual_bug": {
        "title": "Visual Bug (Diálogo Corrompido)",
        "description": "Tags de formatação malfechadas e waypoint comentado.",
        "file_name": "nelly01_visual.lua"
    }
}

WORKLOAD_PROMPTS = {
    "Carga_A": {
        "title": "Geração de Asserções e Testes Unitários em Lua",
        "prompt_template": """[PERSONA]
Você é um Engenheiro de Automação de Testes de Jogos (SDET) especialista na linguagem Lua e na API do jogo Naev (v0.13.5).

[CONTEXTO - CÓDIGO DA MISSÃO]
{code_context}

[TAREFA]
Escreva uma função de teste automatizada em Lua chamada `test_nelly01_inventory_and_equipment()` que realize as seguintes verificações:
1. Verifique se o jogador possui o equipamento "Ion Cannon" instalado (`player.pilot():outfitsList()`).
2. Verifique se a quantidade de minério ("Ore") na carga (`player.pilot():cargoManifest()["Ore"]`) é maior que zero após o embarque na nave derelict.
3. Se qualquer uma das asserções falhar, invoque `error()` exibindo a mensagem do defeito.

[RESTRIÇÕES DE SAÍDA]
- Retorne EXCLUSIVAMENTE o código fonte em Lua dentro de ```lua ... ```.
"""
    },
    "Carga_B": {
        "title": "Planejamento de Smoke Testing e Mapeamento de Stuck Bugs",
        "prompt_template": """[PERSONA]
Você é um Analista de Garantia de Qualidade de Jogos (Game Tester QA) especializado em testes funcionais e Stuck Bugs.

[CONTEXTO - CÓDIGO DA MISSÃO]
{code_context}

[TAREFA]
1. Elabore o plano de Teste de Fumaça (Smoke Test) com o caminho crítico (Golden Path) em exatamente 5 passos sequenciais.
2. Identifique 2 condições de falha de trava (Stuck Bugs) que impediriam o jogador de concluir esta missão.

[RESTRIÇÕES DE SAÍDA]
Responda seguindo a estrutura:

### Plano de Teste de Fumaça (Caminho Crítico)
1. [Passo 1]
2. [Passo 2]
3. [Passo 3]
4. [Passo 4]
5. [Passo 5]

### Mapeamento de Condições de Trava (Stuck Bugs)
- **Risco 1:** [Descrição e localização]
- **Risco 2:** [Descrição e localização]
"""
    },
    "Carga_C": {
        "title": "Triagem Automatizada de Logs de Erro",
        "prompt_template": """[PERSONA]
Você é um Especialista em Triagem de Defeitos de Software (Defect Triage Specialist).

[CONTEXTO - LOG DO CONSOLE / SCRIPT]
{code_context}

[TAREFA]
Examine a execução do script e realize a triagem do defeito:
1. Identifique o arquivo, linha exata e exceção disparada.
2. Classifique o bug (Crash Bug, Stuck Bug, Logical Bug ou Visual Bug).
3. Descreva a causa raiz provável e sugira a correção.

[RESTRIÇÕES DE SAÍDA]
Retorne EXCLUSIVAMENTE um objeto JSON válido no seguinte formato:
{{
  "bug_report": {{
    "environment": "Naev 0.13.5 (Lua Engine)",
    "file": "string",
    "line": 196,
    "bug_category": "Crash Bug | Stuck Bug | Logical Bug | Visual Bug",
    "exception_type": "string",
    "root_cause_analysis": "string",
    "recommended_fix": "string"
  }}
}}
"""
    }
}

# ==============================================================================
# CLASSE PRINCIPAL DO BENCHMARK V12
# ==============================================================================

class NaevQABenchmarkV12:
    def __init__(self, dry_run: bool = False, repetitions: int = 3, target_model: str = "all",
                 fault_dir: str = "fault_scenarios", ollama_timeout: int = 300, gemini_model: str = DEFAULT_GEMINI_MODEL):
        self.dry_run = dry_run
        self.repetitions = repetitions
        self.target_model = target_model.lower()
        self.fault_dir = fault_dir
        self.ollama_timeout = ollama_timeout
        self.gemini_model = gemini_model
        
        # Filtra os modelos de acordo com a flag --model
        self.active_models = {}
        for m_name, m_cfg in ALL_MODELS_CONFIG.items():
            if self.target_model in ("all", m_cfg["key_name"]):
                cfg_copy = dict(m_cfg)
                if cfg_copy["provider"] == "google":
                    cfg_copy["model_id"] = self.gemini_model
                self.active_models[m_name] = cfg_copy
                
        self.results = []

    def load_scenario_code(self, scenario_key: str) -> str:
        scenario_info = FAULT_SCENARIOS[scenario_key]
        file_path = os.path.join(self.fault_dir, scenario_info["file_name"])

        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        
        return f"-- [SCENARIO: {scenario_info['title']}]\n-- {scenario_info['description']}\n-- (Arquivo {file_path} não encontrado localmente, utilizando snippet sintético)\n"

    def call_ollama_api(self, prompt: str, model_id: str, host: str = "http://localhost:11434") -> str:
        url = f"{host}/api/generate"
        payload = {
            "model": model_id,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": 2048}
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        
        with urllib.request.urlopen(req, timeout=self.ollama_timeout) as resp:
            res_data = json.loads(resp.read().decode("utf-8"))
            return res_data.get("response", "")

    def call_google_api(self, prompt: str, model_id: str, api_key: str) -> str:
        time.sleep(1.0)
        domain = '.'.join(['generativelanguage', 'googleapis', 'com'])
        clean_model = model_id.replace("models/", "")
        
        # Tenta endpoints /v1 e /v1beta
        endpoints = [
            f"https://{domain}/v1/models/{clean_model}:generateContent?key={api_key}",
            f"https://{domain}/v1beta/models/{clean_model}:generateContent?key={api_key}"
        ]
        
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.0, "maxOutputTokens": 2048}
        }
        data = json.dumps(payload).encode("utf-8")
        
        last_error = None
        for url in endpoints:
            req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
            for attempt in range(5):
                try:
                    with urllib.request.urlopen(req, timeout=60) as resp:
                        res_data = json.loads(resp.read().decode("utf-8"))
                        return res_data["candidates"][0]["content"]["parts"][0]["text"]
                except urllib.error.HTTPError as e:
                    err_body = e.read().decode("utf-8") if hasattr(e, "read") else str(e)
                    last_error = f"HTTP Error {e.code}: {err_body}"
                    if e.code in (429, 503) and attempt < 4:
                        time.sleep(5.0 * (attempt + 1))
                        continue
                    elif e.code == 404:
                        # Tenta o próximo endpoint
                        break
                    else:
                        raise RuntimeError(last_error)
        raise RuntimeError(last_error or "Falha ao comunicar com a API do Gemini")

    def evaluate_output(self, workload_key: str, output_text: str, expected_scenario: str) -> Dict[str, Any]:
        metrics = {
            "valid_format": False,
            "compilable_or_parseable": False,
            "defect_detected": False,
        }

        if workload_key == "Carga_A":
            has_lua = "```lua" in output_text or "function test_" in output_text
            has_assert = "error(" in output_text or "assert(" in output_text
            metrics["valid_format"] = has_lua
            metrics["compilable_or_parseable"] = has_lua and has_assert
            metrics["defect_detected"] = "outfitsList" in output_text or "cargoManifest" in output_text

        elif workload_key == "Carga_B":
            has_headers = "Plano de Teste" in output_text and "Stuck" in output_text
            has_steps = bool(re.search(r"1\..*2\..*3\..*4\..*5\.", output_text, re.DOTALL))
            metrics["valid_format"] = has_headers
            metrics["compilable_or_parseable"] = has_steps
            metrics["defect_detected"] = "Stuck" in output_text

        elif workload_key == "Carga_C":
            cleaned = re.sub(r"^```json\s*", "", output_text.strip())
            cleaned = re.sub(r"\s*```$", "", cleaned)
            try:
                data = json.loads(cleaned)
                metrics["valid_format"] = "bug_report" in data
                metrics["compilable_or_parseable"] = True
                detected_cat = data.get("bug_report", {}).get("bug_category", "")
                
                if expected_scenario == "crash_bug" and detected_cat == "Crash Bug":
                    metrics["defect_detected"] = True
                elif expected_scenario == "stuck_bug" and detected_cat == "Stuck Bug":
                    metrics["defect_detected"] = True
                elif expected_scenario == "logical_bug" and detected_cat == "Logical Bug":
                    metrics["defect_detected"] = True
                elif expected_scenario == "visual_bug" and detected_cat == "Visual Bug":
                    metrics["defect_detected"] = True
                elif detected_cat in ["Crash Bug", "Stuck Bug", "Logical Bug", "Visual Bug"]:
                    metrics["defect_detected"] = True
            except Exception:
                pass

        return metrics

    def run_benchmark(self):
        print("==================================================================")
        print("  INICIANDO BENCHMARKING DE QA NO NAEV (VERSÃO 12 - FAULT SEEDING)")
        print(f"  Modelos Selecionados: {', '.join(self.active_models.keys())}")
        print(f"  Cenários de Falha: {len(FAULT_SCENARIOS)} ({', '.join(FAULT_SCENARIOS.keys())})")
        print(f"  Ollama Timeout Limit: {self.ollama_timeout}s")
        print(f"  Modo: {'DRY-RUN (Simulado)' if self.dry_run else 'EXECUÇÃO REAL'}")
        print(f"  Repetições: {self.repetitions} por combinação")
        print("==================================================================\n")

        total_runs = len(self.active_models) * len(FAULT_SCENARIOS) * len(WORKLOAD_PROMPTS) * self.repetitions
        run_count = 0

        for model_name, model_info in self.active_models.items():
            for scenario_key, scenario_info in FAULT_SCENARIOS.items():
                scenario_code = self.load_scenario_code(scenario_key)

                for workload_key, workload_info in WORKLOAD_PROMPTS.items():
                    prompt = workload_info["prompt_template"].replace("{code_context}", scenario_code)

                    for rep in range(1, self.repetitions + 1):
                        run_count += 1
                        print(f"[{run_count}/{total_runs}] {model_name} | {scenario_key} | {workload_key} (Rep {rep}/{self.repetitions})...", end="", flush=True)

                        start_time = time.time()
                        if self.dry_run:
                            time.sleep(0.01)
                            raw_output = f"```lua\n-- [DRY RUN] {model_name} response for {scenario_key}\nfunction test_nelly01()\n  assert(player.pilot():outfitsList())\nend\n```"
                        else:
                            provider = model_info["provider"]
                            try:
                                if provider == "ollama":
                                    raw_output = self.call_ollama_api(prompt, model_info["model_id"])
                                elif provider == "google":
                                    api_key = os.environ.get(model_info.get("env_key", "GEMINI_API_KEY"), "")
                                    if not api_key:
                                        raise ValueError("GEMINI_API_KEY não configurada no ambiente.")
                                    raw_output = self.call_google_api(prompt, model_info["model_id"], api_key)
                                else:
                                    raw_output = f"Provedor desconhecido: {provider}"
                            except Exception as e:
                                raw_output = f"ERRO: {str(e)}"

                        elapsed_time = round(time.time() - start_time, 3)
                        metrics = self.evaluate_output(workload_key, raw_output, scenario_key)

                        record = {
                            "timestamp": datetime.now().isoformat(),
                            "run_id": run_count,
                            "model": model_name,
                            "model_id": model_info["model_id"],
                            "fault_scenario": scenario_key,
                            "scenario_title": scenario_info["title"],
                            "workload": workload_key,
                            "workload_title": workload_info["title"],
                            "repetition": rep,
                            "latency_seconds": elapsed_time,
                            "valid_format": metrics["valid_format"],
                            "compilable_or_parseable": metrics["compilable_or_parseable"],
                            "defect_detected": metrics["defect_detected"],
                            "raw_output": raw_output
                        }

                        self.results.append(record)
                        print(f" OK ({elapsed_time}s)")

        return self.results

    def export_results(self, json_path: str, csv_path: str):
        json_dir = os.path.dirname(json_path)
        if json_dir:
            os.makedirs(json_dir, exist_ok=True)
            
        existing_json = []
        if os.path.exists(json_path):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    existing_json = json.load(f)
            except Exception:
                existing_json = []
                
        all_json = existing_json + self.results
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(all_json, f, ensure_ascii=False, indent=2)

        csv_dir = os.path.dirname(csv_path)
        if csv_dir:
            os.makedirs(csv_dir, exist_ok=True)

        write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
        with open(csv_path, "a", encoding="utf-8") as f:
            if write_header:
                f.write("model,model_id,fault_scenario,workload,repetition,latency_seconds,valid_format,compilable_parseable,defect_detected\n")
            for r in self.results:
                f.write(f"{r['model']},{r['model_id']},{r['fault_scenario']},{r['workload']},{r['repetition']},{r['latency_seconds']},{r['valid_format']},{r['compilable_or_parseable']},{r['defect_detected']}\n")

        print(f"\n[SUCESSO] Resultados salvos/anexados em:")
        print(f" - JSON: {json_path}")
        print(f" - CSV:  {csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Runner de Benchmarking de QA para Naev 0.13.5 (v12 - Fault Seeding)")
    parser.add_argument("--real", action="store_true", help="Executa requisições reais")
    parser.add_argument("--model", type=str, default="all", choices=["all", "llama", "qwen", "gemini"], help="Filtra a execução por modelo específico (Padrão: all)")
    parser.add_argument("--runs", type=int, default=3, help="Número de repetições por combinação (Padrão: 3)")
    parser.add_argument("--ollama-timeout", type=int, default=300, help="Timeout em segundos para o Ollama local (Padrão: 300s)")
    parser.add_argument("--gemini-model", type=str, default=DEFAULT_GEMINI_MODEL, help=f"ID do modelo Gemini (Padrão: {DEFAULT_GEMINI_MODEL})")
    parser.add_argument("--fault-dir", type=str, default="fault_scenarios", help="Diretório onde os arquivos de falha estão salvos")
    parser.add_argument("--out-json", type=str, default="benchmark_results_faults.json", help="Arquivo JSON de saída")
    parser.add_argument("--out-csv", type=str, default="benchmark_summary_faults.csv", help="Arquivo CSV de saída")

    args = parser.parse_args()

    bench = NaevQABenchmarkV12(
        dry_run=not args.real,
        repetitions=args.runs,
        target_model=args.model,
        fault_dir=args.fault_dir,
        ollama_timeout=args.ollama_timeout,
        gemini_model=args.gemini_model
    )
    bench.run_benchmark()
    bench.export_results(args.out_json, args.out_csv)
