"""
Streamlit Web UI — Closed-Book Hallucination Risk
-------------------------------------------------

Browser UI:
- Enter/Gemini API key (or rely on env var)
- Pick model, tune evaluation knobs
- Enter prompt, run evaluation
- See decision, Δ̄, B2T, ISR, EDFL RoH bound, next‑step guidance
- Optionally generate an answer (if allowed) and export SLA JSON

Run:
  pip install streamlit google-generativeai>=0.5.0
  streamlit run app/web/web_app.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from dataclasses import asdict

import streamlit as st

# Robust import path resolution for both local and cloud deployment
try:
    # Try standard relative import first (works in local env)
    import scripts.hallucination_toolkit as in_sys
    GeminiBackend = in_sys.GeminiBackend
    OpenRouterBackend = in_sys.OpenRouterBackend
    GeminiItem = in_sys.GeminiItem
    GeminiPlanner = in_sys.GeminiPlanner
    generate_answer_if_allowed = in_sys.generate_answer_if_allowed
    make_sla_certificate = in_sys.make_sla_certificate
except ImportError:
    # Fallback to dynamic path resolution (for cloud deployment)
    current_dir = Path(__file__).parent
    project_root = current_dir.parent

    # Try multiple potential paths
    possible_paths = [
        project_root,
        project_root / "scripts",
        current_dir / ".." / "scripts",
        current_dir / "../../../scripts",
        project_root.parent / "hallucination_bayes" / "scripts",
    ]

    for path_obj in possible_paths:
        path_str = str(path_obj)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

    try:
        from scripts.hallucination_toolkit import (
            GeminiBackend,
            OpenRouterBackend,
            GeminiItem,
            GeminiPlanner,
            generate_answer_if_allowed,
            make_sla_certificate,
        )
    except ImportError:
        # Absolute fallback - this should work locally at least
        import sys
        sys.path.append('scripts')
        from hallucination_toolkit import (
            GeminiBackend,
            OpenRouterBackend,
            GeminiItem,
            GeminiPlanner,
            generate_answer_if_allowed,
            make_sla_certificate,
        )

DEFAULT_MODELS = [
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-1.5-pro",
    "gemini-1.5-flash",
]


def advice_for_metric(decision_answer: bool, roh: float, isr: float, b2t: float) -> list[str]:
    tips: list[str] = []
    if decision_answer:
        if roh <= 0.05:
            tips.append("Low estimated risk. Proceed to answer.")
        elif roh <= 0.20:
            tips.append("Moderate risk. Provide a cautious answer and cite uncertainty.")
        else:
            tips.append("Elevated risk. Consider asking for more context or abstaining.")
        tips.append("Log decision with Δ̄, B2T, ISR, and EDFL RoH bound.")
        tips.append("Optionally generate an answer now and review before sharing.")
    else:
        tips.append("Abstain: the evidence-to-answer margin is insufficient.")
        tips.append("Ask for more context/evidence or simplify the question.")
        tips.append("If evidence exists, switch to evidence_erase skeleton policy.")
        tips.append("Alternatively lower risk targets (smaller h*) only if acceptable.")
    tips.append(f"Diagnostic: ISR={isr:.3f}, B2T={b2t:.3f}, RoH≤{roh:.3f} (EDFL).")
    return tips


def sidebar_controls():
    st.sidebar.header("🤖 Configure AI Provider & Settings")

    # Provider selection - Gemini or OpenRouter
    with st.sidebar.expander("🔑 API & Model", expanded=True):
        st.markdown("**👆 Select Provider First***")

        provider_options = ["GEMINI", "OPENROUTER"]
        provider_idx = st.selectbox(
            "AI Provider",
            options=provider_options,
            index=0,
            help="Choose your preferred AI provider. Gemini = Google native, OpenRouter = multi-provider routing."
        )

        # Provider-specific API key and models
        if provider_idx == "GEMINI":
            api_key = st.text_input(
                "Google Gemini API Key (Required)",
                value="",
                type="password",
                help="Enter your Google Gemini API key. This is not stored or cached.",
                placeholder="AIza...",
            ).strip()
            model_choice = st.selectbox(
                "Gemini Model",
                options=DEFAULT_MODELS + ["Custom…"],
                index=0,
            )
        else:  # OPENROUTER
            api_key = st.text_input(
                "OpenRouter API Key (Required)",
                value="",
                type="password",
                help="Enter your OpenRouter API key. This gives you access to multiple providers via single API.",
                placeholder="sk-or-v1-...",
            ).strip()
            model_choice = st.text_input(
                "OpenRouter Model",
                value="anthropic/claude-3.5-sonnet",
                help="Enter any OpenRouter model name (e.g., anthropic/claude-3.5-sonnet, openai/gpt-4o, google/gemini-flash-1.5-8b)",
            ).strip()

        # Handle custom model input - No need for dropdown since OpenRouter uses text input
        if provider_idx == "GEMINI" and model_choice == "Custom…":
            custom_model = st.text_input("Custom model", value="")
            if custom_model.strip():
                model = custom_model.strip()
            else:
                model = DEFAULT_MODELS[0]
        else:
            model = model_choice

        provider = provider_idx.lower()

    # Other configuration options
    with st.sidebar.expander("Decision thresholds", expanded=False):
        h_star = st.slider("h* (target error when answering)", 0.001, 0.30, 0.05, 0.001)
        isr_threshold = st.slider("ISR threshold", 0.2, 5.0, 1.0, 0.1)
        margin_extra_bits = st.slider("Extra Δ margin (nats)", -1.0, 6.0, 0.0, 0.1)

    with st.sidebar.expander("Advanced", expanded=False):
        skeleton_policy = st.selectbox("Skeleton policy", ["closed_book", "evidence_erase", "auto", "hybrid"], index=0)
        n_samples = st.slider("n_samples (per prompt)", 1, 16, 7)
        m = st.slider("m (skeleton variants)", 1, 16, 6)
        temperature = st.slider("temperature (decision)", 0.0, 1.5, 0.3, 0.025)
        B_clip = st.slider("B_clip", 0.5, 64.0, 12.0, 1.0)
        clip_mode = st.selectbox("clip_mode", ["one-sided", "symmetric"], index=0)

    with st.sidebar.expander("Answer generation", expanded=False):
        want_answer = st.checkbox("Generate answer if allowed", value=False)

    return {
        "provider": provider,
        "api_key": api_key,
        "model": model,
        "n_samples": int(n_samples),
        "m": int(m),
        "skeleton_policy": skeleton_policy,
        "temperature": float(temperature),
        "h_star": float(h_star),
        "isr_threshold": float(isr_threshold),
        "margin_extra_bits": float(margin_extra_bits),
        "B_clip": float(B_clip),
        "clip_mode": clip_mode,
        "want_answer": bool(want_answer),
    }


def main() -> None:
    st.set_page_config(page_title="Hallucination Risk Checker", page_icon="🧪", layout="centered")
    st.title("Closed‑Book Hallucination Risk Checker")
    st.caption("Multi‑provider support; uses EDFL / B2T / ISR to decide answer vs abstain.")

    cfg = sidebar_controls()

    prompt = st.text_area(
        "Enter your prompt",
        height=180,
        placeholder="Ask a question (no external evidence).",
    ).strip()

    col_run, col_reset = st.columns([1, 1])
    run_clicked = col_run.button("Run evaluation", type="primary")
    reset_clicked = col_reset.button("Reset")
    if reset_clicked:
        st.experimental_rerun()

    if run_clicked:
        if not prompt:
            st.warning("Please enter a prompt.")
            return
        provider = cfg["provider"].upper()
        if not cfg["api_key"]:
            if provider == "GEMINI":
                st.error("API key is required. Please enter your Google Gemini API key in the sidebar.")
            else:
                st.error("API key is required. Please enter your OpenRouter API key in the sidebar.")
            return



        item = GeminiItem(
            prompt=prompt,
            n_samples=cfg["n_samples"],
            m=cfg["m"],
            skeleton_policy=cfg["skeleton_policy"],
        )

        try:
            # Dynamically create backend based on selected provider
            if cfg["provider"] == "gemini":
                backend = GeminiBackend(api_key=cfg["api_key"], model=cfg["model"])
                backend_name = "Gemini"
                backend_lib = "`google-generativeai>=0.5.0`"
            elif cfg["provider"] == "openrouter":
                backend = OpenRouterBackend(api_key=cfg["api_key"], model=cfg["model"])
                backend_name = "OpenRouter"
                backend_lib = "`openai`"
            else:
                backend = GeminiBackend(api_key=cfg["api_key"], model=cfg["model"])
                backend_name = "Gemini"
                backend_lib = "`google-generativeai>=0.5.0`"
        except Exception as e:
            st.error(f"Failed to initialize {backend_name} backend: {e}")
            st.info(f"Install {backend_lib} and ensure the API key is valid.")
            return

        planner = GeminiPlanner(
            backend=backend,
            temperature=cfg["temperature"],
        )

        with st.spinner("Evaluating… this can take a moment"):
            metrics = planner.run(
                [item],
                h_star=cfg["h_star"],
                isr_threshold=cfg["isr_threshold"],
                margin_extra_bits=cfg["margin_extra_bits"],
                B_clip=cfg["B_clip"],
                clip_mode=cfg["clip_mode"],
            )

        m = metrics[0]
        decision_str = "Answer" if m.decision_answer else "Abstain"
        if m.decision_answer:
            st.success(f"Decision: {decision_str}")
        else:
            st.warning(f"Decision: {decision_str}")

        st.code(m.rationale, language="text")

        st.subheader("Next steps")
        for tip in advice_for_metric(m.decision_answer, m.roh_bound, m.isr, m.b2t):
            st.markdown(f"- {tip}")

        if cfg["want_answer"]:
            with st.spinner("Generating answer…"):
                ans = generate_answer_if_allowed(backend, item, m, max_tokens_answer=256)
            if ans:
                st.subheader("Model answer")
                st.write(ans)
            else:
                st.info("No answer generated (model abstained or error).")

        report = planner.aggregate([item], metrics, h_star=cfg["h_star"], isr_threshold=cfg["isr_threshold"], margin_extra_bits=cfg["margin_extra_bits"])  # type: ignore
        cert = make_sla_certificate(report, model_name=cfg["model"], confidence_1_minus_alpha=0.95)
        cert_json = json.dumps(asdict(cert), indent=2)
        st.download_button(
            "Download SLA certificate (JSON)",
            data=cert_json,
            file_name="sla_certificate.json",
            mime="application/json",
        )

        st.caption("All information measures are in nats. Closed‑book uses semantic masking.")


if __name__ == "__main__":
    main()
# Copyright (c) 2024 Hassana Labs
# Licensed under the MIT License - see LICENSE file for details
