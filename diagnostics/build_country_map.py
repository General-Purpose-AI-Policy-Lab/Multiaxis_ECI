"""Country-of-origin map for every model_version in the processed data.

Three buckets only: US, CN, Other. Method chain, in priority order:
  1. `data/curated/model_country_overrides.csv` (hand-reviewed) always wins.
  2. `organization` column, lead org (first comma-separated name) looked up
     in ORG_COUNTRY. A row whose organization is missing or the literal
     string "other" (Epoch's own "don't know" value) skips to step 3.
  3. NAME_PREFIX: substring match on model_version, longest key first so a
     specific name (e.g. "llava-v1.6-vicuna-13b") beats a shorter generic
     one (e.g. "vicuna") that happens to also match.
  4. Unresolved -> country=Other, method=unresolved, noted for review.

Output: data/curated/model_country.csv (model_version,country,organization,
method,notes), one row per unique model_version, sorted by country then name.

Never hand-edit the output. Add corrections to the overrides CSV instead.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data/processed/benchmarks_merged.csv"
OVERRIDES_PATH = ROOT / "data/curated/model_country_overrides.csv"
OUT_PATH = ROOT / "data/curated/model_country.csv"

# Lead organization (first name in a comma-separated organization string) ->
# country. Every distinct lead org present in the data must be a key here;
# an unmapped one fails the build loudly rather than silently guessing.
ORG_COUNTRY = {
    # ---- US ----
    "OpenAI": "US", "Anthropic": "US", "Google DeepMind": "US",
    "Google Research": "US", "Meta AI": "US", "xAI": "US",
    "Microsoft": "US", "Microsoft Research": "US", "NVIDIA": "US",
    "Amazon": "US", "Thinking Machines": "US", "thinkingmachines": "US",
    "Cognition": "US", "Databricks": "US", "MosaicML": "US",
    "Poolside": "US", "Perplexity": "US", "Cursor": "US",
    "EleutherAI": "US", "Salesforce": "US", "Salesforce Research": "US",
    "Allen Institute for AI": "US", "IBM": "US", "Inflection AI": "US",
    "Cerebras Systems": "US", "Inception Labs": "US", "Deep Cogito": "US",
    "Nous Research": "US", "Arcee AI": "US", "Hugging Face": "US",
    # ---- CN ----
    "Alibaba": "CN", "DeepSeek": "CN", "Moonshot": "CN",
    "Z.ai (Zhipu AI)": "CN", "01.AI": "CN", "MiniMax": "CN",
    "Xiaomi Corp": "CN", "Baichuan": "CN", "StepFun": "CN",
    "ByteDance": "CN", "Shanghai AI Lab": "CN", "Ant Group": "CN",
    "Butterfly Effect (Monica)": "CN", "Tsinghua University": "CN",
    # ---- Other ----
    "Mistral AI": "Other", "Technology Innovation Institute": "Other",
    "Cohere": "Other", "Upstage": "Other", "Reka AI": "Other",
    "Stability AI": "Other", "Large Model Systems Organization": "Other",
    "Prime Intellect": "Other",
}

# organization values that carry no real vendor info, so a row with one of
# these falls through to the name-prefix rules instead of ORG_COUNTRY.
ORG_UNKNOWN_VALUES = {"other"}

# Substring rules for models with no usable organization. Checked longest
# key first, so a specific string always beats a shorter generic one it
# happens to contain (e.g. "instructblip" before "vicuna", "llava-v1.6-
# vicuna-13b" before both). Keys are matched case-insensitively.
#
# Sourced from the vendor-prefix worked examples plus WebSearch for the
# genuinely obscure ones (see build report / commit message for citations); a few
# (LinVT, ml-elephant) turned up no reliable affiliation and are left
# unresolved on purpose rather than guessed.
NAME_PREFIX = {
    # ---- CN ----
    "qwen": "CN", "internlm": "CN", "chatglm": "CN", "glm": "CN",
    "kimi": "CN", "minimax": "CN", "internvl": "CN", "mplug": "CN",
    "vita": "CN", "video-ccam": "CN", "sharegpt4video": "CN",
    "yi-": "CN", "baichuan": "CN", "deepseek": "CN", "minicpm": "CN",
    "mimo": "CN",  # Xiaomi's MiMo model line
    "videollama": "CN", "chat-univi": "CN", "chat-uni-vi": "CN",
    "kangaroo": "CN",  # Meituan + UCAS Beijing (arXiv:2408.15542)
    "xiaoyi": "CN",  # Huawei Xiaoyi Deep Research
    "bytevideollm": "CN",  # ByteDance (github Hon-Wong/ByteVideoLLM)
    "st-llm": "CN",  # TencentARC
    "slime": "CN",  # UCAS + Alibaba (arXiv:2406.08487)
    "oryx": "CN",  # Tsinghua-led, w/ Tencent + NTU (arXiv:2409.12961)
    "timemarker": "CN",  # Meituan Inc (arXiv:2411.18211)
    "vilamp": "CN",  # Renmin University of China (arXiv:2504.02438)
    "video-xl": "CN",  # Shanghai Jiao Tong-led (arXiv:2409.14485)
    "video_chat2": "CN",  # OpenGVLab / Shanghai AI Lab
    "video-llava": "CN",  # PKU-YuanGroup, Peking University
    "long-llava": "CN",  # HKUST(GZ)/CUHK-SZ (LongLLaVA, arXiv:2409.02889)
    "llama3-llava-next": "CN",  # lmms-lab continuation, ByteDance-led
    "llava-video": "CN",  # lmms-lab, ByteDance-led (LLaVA-OneVision team)
    "llava-onevision": "CN",  # first author ByteDance (arXiv:2408.03326)
    "longva": "CN",  # same lmms-lab/ByteDance team as LLaVA-OneVision
    # ---- US ----
    "grok": "US", "gemini": "US", "gemma": "US", "gpt": "US",
    "codex": "US", "o1": "US", "o3": "US", "palm": "US", "t5": "US",
    "switch": "US", "phi": "US", "learnlm": "US",
    "computer-use-preview": "US", "sonar": "US",
    "mm1": "US",  # Apple (arXiv:2403.09611)
    "openhands-lm": "US",  # All Hands AI, Boston MA
    "dracarys2": "US",  # Abacus.AI
    "redpajama": "US",  # Together AI
    "open_llama": "US",  # OpenLM Research community project
    "opt-13b": "US",  # Meta OPT
    "text-davinci": "US",  # OpenAI
    "instructblip": "US",  # Salesforce Research
    "granite": "US",  # IBM
    "lfm": "US",  # Liquid AI (Liquid Foundation Models), Boston MA
    # llava-v1.5/v1.6: original Haotian Liu / Microsoft Research + UW
    # Madison release, distinct from the later ByteDance-led lmms-lab
    # extensions (LLaVA-OneVision/Video/LongVA) above.
    "llava-v1.5-7b": "US", "llava-v1.6-mistral-7b": "US",
    "llava-v1.6-vicuna-13b": "US", "llava-v1.6-vicuna-7b": "US",
    # ---- Other ----
    "falcon": "Other",  # Technology Innovation Institute (AE)
    "stablelm": "Other",  # Stability AI
    "vicuna": "Other",  # LMSYS (Large Model Systems Organization)
    "c4ai": "Other",  # Cohere
    "mistral-medium": "Other", "mistral-small": "Other",  # Mistral AI (FR)
    "tiny-recursion-model": "Other",  # Samsung SAIL Montreal (Korea HQ)
    "aria": "Other",  # Rhymes AI, Tokyo-based (per InfoQ/Decrypt reporting)
    "livecc": "Other",  # NUS Show Lab-led, w/ ByteDance (arXiv:2504.16030)
}


def lead_org(org):
    """First name in a comma-separated organization string, or None."""
    if not isinstance(org, str) or not org.strip():
        return None
    first = org.split(",")[0].strip()
    return None if first.lower() in ORG_UNKNOWN_VALUES else first


def classify_by_name(model_version):
    """Longest-substring-first match against NAME_PREFIX. None if no hit."""
    mv_lower = model_version.lower()
    for key in sorted(NAME_PREFIX, key=len, reverse=True):
        if key in mv_lower:
            return NAME_PREFIX[key], key
    return None, None


def classify(model_version, organization):
    org = lead_org(organization)
    if org is not None:
        if org not in ORG_COUNTRY:
            raise ValueError(
                f"Unmapped lead organization {org!r} (from {organization!r}, "
                f"model {model_version!r}). Add it to ORG_COUNTRY."
            )
        return ORG_COUNTRY[org], "org_map", f"lead org: {org}"

    country, key = classify_by_name(model_version)
    if country is not None:
        return country, "prefix", f"name-prefix match: {key!r}"

    return "Other", "unresolved", "no organization, no matching name prefix"


def main():
    df = pd.read_csv(DATA_PATH)
    base = df[["model_version", "organization"]].drop_duplicates("model_version")

    rows = []
    for mv, org in zip(base.model_version, base.organization):
        country, method, notes = classify(mv, org)
        rows.append(dict(model_version=mv, country=country,
                         organization=org if isinstance(org, str) else "",
                         method=method, notes=notes))
    result = pd.DataFrame(rows)

    if OVERRIDES_PATH.exists():
        overrides = pd.read_csv(OVERRIDES_PATH)
        if len(overrides):
            override_map = overrides.set_index("model_version")
            known = set(result.model_version)
            for mv in override_map.index:
                if mv not in known:
                    raise ValueError(
                        f"Override for {mv!r} does not match any "
                        f"model_version in the processed data."
                    )
            result = result.set_index("model_version")
            for mv, row in override_map.iterrows():
                result.loc[mv, "country"] = row["country"]
                result.loc[mv, "method"] = "override"
                result.loc[mv, "notes"] = row.get("notes", "")
            result = result.reset_index()

    result = result.sort_values(["country", "model_version"]).reset_index(drop=True)
    result = result[["model_version", "country", "organization", "method", "notes"]]

    # every unique model_version gets exactly one row, in one of three buckets
    assert set(result.model_version) == set(base.model_version)
    assert result.model_version.is_unique
    assert set(result.country) <= {"US", "CN", "Other"}

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT_PATH, index=False)

    print(f"wrote {OUT_PATH} ({len(result)} models)")
    print("\nper country:")
    print(result.country.value_counts().to_string())
    print("\nper method:")
    print(result.method.value_counts().to_string())


if __name__ == "__main__":
    main()
