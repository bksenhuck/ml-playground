"""Rule-based fallback insight engine.

Generates natural-language insights from experiment runs without calling any
external API. Used whenever Gemini is unavailable or returns an error.

Design principles:
  - Analyse the DataFrame systematically (best runs, averages, config patterns)
  - Route to a focused response function based on question keywords
  - Always produce a coherent paragraph — never a raw bullet dump
  - Respond in Brazilian Portuguese to match the rest of the UI
"""
from __future__ import annotations

import pandas as pd

# ── Constants ─────────────────────────────────────────────────────────────────

_METRIC_COLS = [
    "metric_accuracy",
    "metric_precision",
    "metric_recall",
    "metric_f1",
    "metric_roc_auc",
]

# Human-readable metric labels (Portuguese)
_METRIC_LABELS: dict[str, str] = {
    "metric_accuracy":  "acurácia",
    "metric_precision": "precisão",
    "metric_recall":    "recall",
    "metric_f1":        "F1",
    "metric_roc_auc":   "ROC AUC",
}

_GOOD  = 0.80   # threshold for "good" performance
_POOR  = 0.65   # threshold for "poor" performance


# ── Public entry point ────────────────────────────────────────────────────────

def generate_fallback_insight(question: str, runs_df: pd.DataFrame) -> str:
    """Generate a heuristic insight paragraph for the given question and runs.

    Args:
        question: The user's free-text question (keyword detection only).
        runs_df:  DataFrame of experiment runs — already trimmed to ≤5 rows.

    Returns:
        A natural-language paragraph in Brazilian Portuguese. Never empty.
    """
    if runs_df is None or runs_df.empty:
        return (
            "Nenhum experimento disponível para análise. "
            "Execute ao menos um experimento e selecione-o na tabela."
        )

    analysis = _analyze_runs(runs_df)
    q = question.lower()

    # Route to the most relevant response based on question keywords
    if _hits(q, ["melhor", "top", "maior", "best", "qual", "ganhou", "venceu"]):
        return _best_run_response(analysis)
    if _hits(q, ["recall", "falso negativo", "sensibilidade", "missed"]):
        return _recall_response(analysis)
    if _hits(q, ["precisão", "precision", "falso positivo"]):
        return _precision_response(analysis)
    if _hits(q, ["roc", "auc", "discrimin"]):
        return _roc_response(analysis)
    if _hits(q, ["comparar", "comparação", "vs", "diferença", "modelo", "model"]):
        return _comparison_response(analysis)
    if _hits(q, ["próximo", "proximo", "sugerir", "recomendar", "melhorar", "otimizar", "next"]):
        return _next_step_response(analysis)
    if _hits(q, ["overfitting", "overfit", "generaliz"]):
        return _overfitting_response(analysis)

    # Default: holistic overview
    return _general_response(analysis)


# ── Internal analysis ─────────────────────────────────────────────────────────

def _analyze_runs(df: pd.DataFrame) -> dict:
    """Extract key statistics from the runs DataFrame."""
    available = [c for c in _METRIC_COLS if c in df.columns]

    # Best run per metric (by row index)
    best: dict[str, dict] = {}
    for col in available:
        numeric = pd.to_numeric(df[col], errors="coerce")
        idx = numeric.idxmax()
        if pd.notna(idx):
            row = df.loc[idx]
            best[col] = {
                "name":  str(row.get("run_name") or row.get("name") or f"run-{idx}"),
                "model": str(row.get("model", "?")),
                "value": float(numeric[idx]),
            }

    # Column-wise averages
    avg: dict[str, float] = {
        col: float(pd.to_numeric(df[col], errors="coerce").mean())
        for col in available
    }

    # Model distribution
    models: dict[str, int] = (
        df["model"].value_counts().to_dict() if "model" in df.columns else {}
    )

    # Config flags
    any_balanced = (
        "class_weight" in df.columns and (df["class_weight"] == "balanced").any()
    )
    has_poly = (
        "poly_features" in df.columns
        and df["poly_features"].astype(str).str.lower().eq("true").any()
    )
    scaling_used = (
        "scaling" in df.columns and df["scaling"].ne("none").any()
    )

    return {
        "n":                len(df),
        "available_metrics": available,
        "best":             best,
        "avg":              avg,
        "models":           models,
        "any_balanced":     any_balanced,
        "has_poly":         has_poly,
        "scaling_used":     scaling_used,
        "df":               df,
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _hits(text: str, keywords: list[str]) -> bool:
    return any(kw in text for kw in keywords)

def _lbl(col: str) -> str:
    return _METRIC_LABELS.get(col, col)

def _f(v: float) -> str:
    return f"{v:.3f}"


# ── Response generators ───────────────────────────────────────────────────────

def _best_run_response(a: dict) -> str:
    best = a["best"]
    if not best:
        return (
            "Não foi possível identificar o melhor experimento — "
            "verifique se as métricas foram registradas corretamente."
        )

    # F1 is the primary ranking metric (balances precision × recall)
    primary = best.get("metric_f1") or next(iter(best.values()))
    parts = [
        f"O experimento com melhor F1 é '{primary['name']}' "
        f"(modelo: {primary['model']}, F1 = {_f(primary['value'])})."
    ]

    # Mention runners-up on other key metrics
    for col in ["metric_roc_auc", "metric_recall", "metric_accuracy"]:
        b = best.get(col)
        if b and b["name"] != primary["name"]:
            parts.append(
                f"Para {_lbl(col)}, o destaque foi '{b['name']}' ({_f(b['value'])})."
            )

    avg_f1 = a["avg"].get("metric_f1", 1.0)
    if avg_f1 < _POOR:
        parts.append(
            "No geral, os resultados têm espaço para melhora — "
            "considere engenharia de features ou ajuste de hiperparâmetros."
        )

    return " ".join(parts)


def _recall_response(a: dict) -> str:
    avg_recall = a["avg"].get("metric_recall")
    best_recall = a["best"].get("metric_recall")
    parts = []

    if avg_recall is not None:
        if avg_recall < _POOR:
            parts.append(
                f"O recall médio está baixo ({_f(avg_recall)}), indicando que o modelo "
                "perde muitos casos positivos (falsos negativos)."
            )
            if not a["any_balanced"]:
                parts.append(
                    "Isso frequentemente ocorre com classes desbalanceadas. "
                    "Tente ativar class_weight=balanced para penalizar erros na classe minoritária."
                )
            parts.append(
                "Outra estratégia: reduza o threshold de classificação de 0.5 para ~0.3–0.4 "
                "para capturar mais positivos — avalie o trade-off com a precisão."
            )
        else:
            parts.append(
                f"O recall médio está satisfatório ({_f(avg_recall)}). "
                "Se ainda houver margem, tente aumentar dados de treino ou explorar "
                "features que capturem melhor os casos positivos."
            )

    if best_recall:
        parts.append(
            f"Melhor recall nos experimentos: '{best_recall['name']}' "
            f"({_f(best_recall['value'])}) usando {best_recall['model']}."
        )

    return " ".join(parts) or "Dados insuficientes para análise de recall."


def _precision_response(a: dict) -> str:
    avg_prec = a["avg"].get("metric_precision")
    best_prec = a["best"].get("metric_precision")
    parts = []

    if avg_prec is not None:
        if avg_prec < _POOR:
            parts.append(
                f"A precisão média está baixa ({_f(avg_prec)}), "
                "indicando alto número de falsos positivos."
            )
            parts.append(
                "Para melhorar: aumente o threshold de classificação (0.6–0.7), "
                "adicione features discriminativas ou remova ruído dos dados de treino."
            )
        else:
            parts.append(
                f"A precisão média está boa ({_f(avg_prec)}). "
                "Verifique se o recall está alinhado para não perder o equilíbrio entre as métricas."
            )

    if best_prec:
        parts.append(
            f"Melhor precisão: '{best_prec['name']}' com {_f(best_prec['value'])}."
        )

    return " ".join(parts) or "Dados insuficientes para análise de precisão."


def _roc_response(a: dict) -> str:
    avg_auc = a["avg"].get("metric_roc_auc")
    best_auc = a["best"].get("metric_roc_auc")
    parts = []

    if avg_auc is not None:
        if avg_auc < 0.70:
            parts.append(
                f"ROC AUC médio baixo ({_f(avg_auc)}): o modelo mal supera um classificador aleatório. "
                "Isso geralmente indica features pouco informativas ou dados muito ruidosos."
            )
            parts.append(
                "Recomendações: explore novas features, aplique transformações (log, interações polinomiais) "
                "ou experimente o Random Forest, que captura relações não-lineares."
            )
        elif avg_auc < _GOOD:
            parts.append(
                f"ROC AUC moderado ({_f(avg_auc)}). "
                "Há espaço para melhora via feature engineering ou otimização de hiperparâmetros."
            )
        else:
            parts.append(
                f"Excelente ROC AUC ({_f(avg_auc)}): capacidade discriminativa elevada. "
                "Certifique-se de que não há data leakage entre treino e teste."
            )

    if best_auc:
        parts.append(
            f"Melhor AUC: '{best_auc['name']}' com {_f(best_auc['value'])}."
        )

    return " ".join(parts) or "Dados insuficientes para análise de ROC AUC."


def _comparison_response(a: dict) -> str:
    models = a["models"]
    df = a["df"]

    if len(models) < 2:
        return (
            "Apenas um tipo de modelo nos experimentos selecionados. "
            "Adicione experimentos com outros modelos (ex: Random Forest vs Logistic Regression) "
            "para uma comparação significativa."
        )

    # Build per-model metric summary
    model_summaries = []
    for model_name in models:
        m_df = df[df["model"] == model_name]
        metric_parts = []
        for col in ["metric_f1", "metric_roc_auc", "metric_accuracy"]:
            if col in m_df.columns:
                avg = pd.to_numeric(m_df[col], errors="coerce").mean()
                if pd.notna(avg):
                    metric_parts.append(f"{_lbl(col)}={_f(avg)}")
        summary = f"{model_name}: {', '.join(metric_parts)}" if metric_parts else model_name
        model_summaries.append(summary)

    response = "Comparação entre modelos — " + " | ".join(model_summaries) + ". "

    best_f1 = a["best"].get("metric_f1")
    if best_f1:
        response += (
            f"O experimento com melhor F1 geral é '{best_f1['name']}' "
            f"({best_f1['model']}, {_f(best_f1['value'])}). "
        )

    avg_f1 = a["avg"].get("metric_f1", 0.0)
    if avg_f1 < _POOR:
        response += "Ambos os modelos têm margem de melhora — considere engenharia de features antes de mais ajustes."
    else:
        response += "Continue iterando sobre o modelo que obteve melhores resultados."

    return response


def _next_step_response(a: dict) -> str:
    suggestions: list[str] = []
    avg_recall = a["avg"].get("metric_recall", 1.0)
    avg_auc    = a["avg"].get("metric_roc_auc", 1.0)
    avg_f1     = a["avg"].get("metric_f1", 1.0)

    # Class imbalance signal: recall is low but balanced weighting not tried
    if avg_recall < _POOR and not a["any_balanced"]:
        suggestions.append(
            "ative class_weight=balanced para melhorar recall em classes desbalanceadas"
        )
    # Weak discrimination → suggest more expressive features
    if avg_auc < 0.75:
        suggestions.append(
            "experimente features adicionais ou interações polinomiais para capturar relações não-lineares"
        )
    # Overall weak results → hyperparameter search
    if avg_f1 < _POOR:
        suggestions.append(
            "ajuste os hiperparâmetros do melhor modelo atual (grid search ou random search)"
        )
    # Scaling missing for logreg
    if not a["scaling_used"] and "logreg" in a["models"]:
        suggestions.append(
            "ative StandardScaler — essencial para Logistic Regression convergir bem"
        )
    # Polynomial features not tried yet
    if not a["has_poly"] and avg_f1 < _GOOD:
        suggestions.append(
            "teste features polinomiais de grau 2 para capturar interações entre variáveis"
        )

    if not suggestions:
        suggestions.append(
            "os resultados já estão sólidos — valide em dados externos ou documente o modelo atual"
        )

    intro = f"Com base nos {a['n']} experimento(s) selecionado(s), as próximas ações recomendadas são: "
    return intro + "; ".join(suggestions) + "."


def _overfitting_response(a: dict) -> str:
    # Without stored train metrics we reason about structural risk only
    parts = [
        "Sem métricas de treino registradas, não é possível confirmar overfitting diretamente. "
        "No entanto, observe estes sinais indiretos:"
    ]

    avg_f1 = a["avg"].get("metric_f1", 0.0)
    if avg_f1 > _GOOD:
        parts.append(
            f"F1 de teste alto ({_f(avg_f1)}) é um bom sinal, "
            "mas confirme com validação cruzada (k-fold)."
        )

    if "rf" in a["models"]:
        parts.append(
            "Random Forest com profundidade alta tende a overfit — "
            "reduza max_depth ou aumente min_samples_leaf."
        )

    parts.append(
        "Regra prática: se accuracy de treino − accuracy de teste > 10 pp, investigue overfitting."
    )

    return " ".join(parts)


def _general_response(a: dict) -> str:
    parts = [f"Análise de {a['n']} experimento(s) selecionado(s)."]

    f1  = a["avg"].get("metric_f1")
    auc = a["avg"].get("metric_roc_auc")

    if f1 is not None:
        level = "boa" if f1 >= _GOOD else "moderada" if f1 >= _POOR else "baixa"
        auc_str = _f(auc) if auc is not None else "n/a"
        parts.append(
            f"Performance geral {level}: F1 médio = {_f(f1)}, ROC AUC médio = {auc_str}."
        )

    best_f1 = a["best"].get("metric_f1")
    if best_f1:
        parts.append(
            f"Destaque: '{best_f1['name']}' ({best_f1['model']}) com F1 = {_f(best_f1['value'])}."
        )

    # Actionable suggestion proportional to how far from "good" we are
    if f1 is not None:
        if f1 < _POOR:
            hint = (
                "Sugestão imediata: ative class_weight=balanced e compare os resultados."
                if not a["any_balanced"]
                else "Sugestão: explore features adicionais ou experimente Random Forest."
            )
        elif f1 < _GOOD:
            hint = "Sugestão: ajuste hiperparâmetros do melhor modelo ou adicione features polinomiais."
        else:
            hint = "Os resultados estão sólidos. Considere validação cruzada para confirmar."
        parts.append(hint)

    return " ".join(parts)
